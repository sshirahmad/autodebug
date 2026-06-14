"""Shared infrastructure: budget tracking, middleware, and model factory.

Each agent is built with LangChain's `create_agent` — see autodebug/agents/{repro,bisect,...}.py.
The agent's tool-execution loop is handled by create_agent; budgets are enforced
via the `budget_middleware` (a pair of before_model/after_model hooks).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentState,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    TodoListMiddleware,
    SummarizationMiddleware,
    after_model,
    before_model,
)
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime

load_dotenv()

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_PROVIDER = "anthropic"


class BudgetExceeded(Exception):
    """Raised by budget_middleware when any limit is hit."""


@dataclass
class Budget:
    """Time + cost budget. Tokens are counted (so we can compute cost and
    populate state.total_tokens) but never used as a limit."""

    time_seconds: int | None = None
    cost_usd: float | None = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    tokens_used: int = field(default=0, init=False)
    cost_used: float = field(default=0.0, init=False)
    calls: int = field(default=0, init=False)  # number of model calls made
    _start: float = field(default=0.0, init=False, repr=False)
    _deadline: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._start = time.monotonic()
        self._deadline = (self._start + self.time_seconds) if self.time_seconds else None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start

    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> float:
        self.tokens_used += input_tokens + output_tokens
        cost = (
            input_tokens * self.cost_per_1k_input / 1000
            + output_tokens * self.cost_per_1k_output / 1000
        )
        self.cost_used += cost
        return cost

    def check(self) -> None:
        if self._deadline and time.monotonic() > self._deadline:
            raise BudgetExceeded(
                f"Time budget exceeded ({self.elapsed_seconds:.1f}s > {self.time_seconds}s; "
                f"{self.tokens_used} tokens, ${self.cost_used:.4f})"
            )
        if self.cost_usd is not None and self.cost_used > self.cost_usd:
            raise BudgetExceeded(
                f"Cost budget exceeded (${self.cost_used:.4f} > ${self.cost_usd}; "
                f"{self.elapsed_seconds:.1f}s, {self.tokens_used} tokens)"
            )

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(
            time_seconds=cfg.time_budget_seconds,
            cost_usd=cfg.cost_budget_usd,
            cost_per_1k_input=cfg.cost_per_1k_input_tokens or 0.0,
            cost_per_1k_output=cfg.cost_per_1k_output_tokens or 0.0,
        )


def budget_middleware(budget: Budget) -> list:
    """Two-piece middleware: check budget before each model call, track tokens after."""

    @before_model
    def _check_budget(state: AgentState, runtime: Runtime) -> None:
        budget.check()
        return None

    @after_model
    def _track_tokens(state: AgentState, runtime: Runtime) -> None:
        budget.calls += 1
        msg = state["messages"][-1]
        meta = getattr(msg, "usage_metadata", None) or {}
        budget.add_tokens(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        return None

    return [_check_budget, _track_tokens]


def session_budget_middleware(state, max_cost: float | None, max_seconds: int | None) -> list:
    """Global ceiling across an orchestrated session (manager + all sub-agents).

    Per-agent budgets only bound a single sub-agent run; a manager that loops
    through REVISING can rack up many full sub-agent runs (black-1 hit ~$31).
    This before_model hook runs on each *manager* turn and inspects the cumulative
    `state.total_cost` (sub-agent costs are added to it as each sub-agent finishes)
    plus wall-clock time, raising BudgetExceeded once either ceiling is crossed —
    which stops the manager from delegating further.
    """
    start = time.monotonic()

    # NB: the hook arg is the agent's graph state — deliberately *not* named
    # `state` so it doesn't shadow the captured DebugState we're measuring.
    @before_model
    def _check_session(_graph_state: AgentState, runtime: Runtime) -> None:
        if max_cost is not None and state.total_cost > max_cost:
            raise BudgetExceeded(
                f"Session cost budget exceeded (${state.total_cost:.4f} > ${max_cost})"
            )
        if max_seconds is not None and (time.monotonic() - start) > max_seconds:
            raise BudgetExceeded(
                f"Session time budget exceeded ({time.monotonic() - start:.0f}s > {max_seconds}s)"
            )
        return None

    return [_check_session]


def tool_call_limit_middleware(limits: dict[str, int]) -> list:
    """Cap how many times each named tool can be called in a single agent run.

    When a tool's limit is reached, further calls to that tool are blocked
    (the model gets a notice) but the agent continues — so it can fall back
    to a different tool. Force the fix agent to attempt `apply_patch` instead
    of endlessly re-reading the same files, force bisect to submit after
    enough exploration, etc.
    """
    return [
        ToolCallLimitMiddleware(
            tool_name=name, run_limit=limit, exit_behavior="continue",
        )
        for name, limit in (limits or {}).items()
    ]


def model_retry_middleware() -> list:
    """Retry transient model-call failures with exponential backoff.

    Provider hiccups (HTTP 400 "provider returned error", 429 rate limits, 5xx,
    timeouts) are common on long runs. Without this, a single failed model call
    propagates out of agent.invoke and aborts the entire pipeline — we saw a
    27-minute run discarded by one 400. After retries are exhausted, `on_failure
    ='continue'` returns an error AIMessage instead of raising, so the agent
    degrades rather than crashing.
    """
    return [ModelRetryMiddleware(max_retries=3, on_failure="continue")]


def planning_middleware() -> list:
    """TodoListMiddleware: a `write_todos` tool so the agent can plan multi-step work.

    The agent maintains a structured task list (plan, mark in-progress, complete),
    which helps long multi-tool runs (bisect, fix) stay on track instead of looping.
    """
    return [TodoListMiddleware()]


# Absolute token trigger for summarization. A fractional trigger needs per-model
# profile metadata (max_input_tokens) that isn't reliably available, so we use an
# absolute count that fits comfortably under a 200k-context model and still leaves
# room for the summary + continued work. Override via env if needed.
_SUMMARIZE_TRIGGER_TOKENS = int(os.getenv("AUTODEBUG_SUMMARIZE_TRIGGER_TOKENS", "120000"))
_SUMMARIZE_KEEP_MESSAGES = int(os.getenv("AUTODEBUG_SUMMARIZE_KEEP_MESSAGES", "20"))


def summarization_middleware(
    model_id: str | None = None, provider: str | None = None
) -> list:
    """SummarizationMiddleware: compress old turns when context grows large.

    Once the running history exceeds the token trigger, older messages are
    summarized while the most recent turns are kept verbatim (AI/Tool pairs
    preserved). This keeps long agent runs from blowing the context window — a
    failure mode behind stalled fix attempts.
    """
    model = build_model(model_id=model_id, provider=provider)
    return [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", _SUMMARIZE_TRIGGER_TOKENS),
            keep=("messages", _SUMMARIZE_KEEP_MESSAGES),
        )
    ]


def require_tool_calls_middleware() -> list:
    """Reject text-only AI responses and force the model to retry with a tool call.

    Modern reasoning models will happily emit thousands of tokens of analysis
    text without ever calling a tool — burning budget without making progress.
    After every model call, if the response has no tool_calls, we append a
    short corrective message and jump back to the model node so it tries
    again. The budget middleware naturally bounds the loop.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    @after_model(can_jump_to=["model"])
    def _require_tool(state: AgentState, runtime: Runtime):
        msg = state["messages"][-1]
        if not isinstance(msg, AIMessage):
            return None
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            return None
        return {
            "messages": [
                HumanMessage(content=(
                    "Your last response had no tool call. Every response MUST "
                    "be a tool call. Make a tool call now — do not write text."
                ))
            ],
            "jump_to": "model",
        }

    return [_require_tool]


# Cap how much trajectory we feed the optimizer — long fix runs can be 100+
# messages and the optimizer sends them all to an LLM. Keep the opening (task
# framing) plus the most recent turns, where the failure actually shows up.
_TRAJ_HEAD = 4
_TRAJ_TAIL = 56


def retry_feedback(goal: str) -> str:
    """Feedback string handed to the optimizer describing why a retry happened."""
    return (
        f"The previous attempt FAILED: it exhausted its time/cost budget before "
        f"{goal}. Study the trajectory for wasted effort — repeated file reads, "
        f"loops, unproductive tangents, or missing tool calls — and revise the "
        f"system prompt so the next attempt avoids those failure modes and reaches "
        f"its submission faster. Preserve the prompt's required tools, output "
        f"contract, and any STOP/format rules."
    )


def attempt_trajectory(agent, config) -> list:
    """Recover the message history of an attempt from its checkpointer.

    Works even when the attempt ended by raising BudgetExceeded — the last
    completed super-step is still persisted to the in-memory checkpoint.
    """
    try:
        return list(agent.get_state(config).values.get("messages", []))
    except Exception:
        return []


def _trim_trajectory(messages: list) -> list:
    if len(messages) <= _TRAJ_HEAD + _TRAJ_TAIL:
        return messages
    return messages[:_TRAJ_HEAD] + messages[-_TRAJ_TAIL:]


def maybe_optimize_prompt(
    current_prompt: str,
    messages: list,
    feedback: str,
    *,
    model_id: str | None = None,
    provider: str | None = None,
) -> str:
    """Improve `current_prompt` from a failed attempt's trajectory via LangMem.

    Uses the gradient prompt optimizer: it reflects on the trajectory + feedback
    and rewrites the prompt to avoid the observed failure mode. Best-effort — any
    error (missing dep, optimizer failure, empty trajectory) returns the original
    prompt so the retry loop is never blocked. Disable with AUTODEBUG_PROMPT_OPTIM=0.
    """
    if os.getenv("AUTODEBUG_PROMPT_OPTIM", "1") == "0" or not messages:
        return current_prompt
    try:
        from langmem import create_prompt_optimizer
        from langmem.prompts.types import AnnotatedTrajectory

        model = build_model(model_id=model_id, provider=provider)
        optimizer = create_prompt_optimizer(model, kind="gradient")
        trajectory = AnnotatedTrajectory(
            messages=_trim_trajectory(messages), feedback=feedback
        )
        improved = optimizer.invoke(
            {"trajectories": [trajectory], "prompt": current_prompt}
        )
        if isinstance(improved, str) and improved.strip():
            return improved
    except Exception:
        pass
    return current_prompt


def submission_middleware(channel: str) -> list:
    """Stop the agent loop as soon as a submit_* tool writes its result `channel`.

    Without this, the agent keeps making LLM calls (and burning budget) after
    its final tool answer is already submitted — verifying its candidate,
    re-reading code, or re-submitting the same answer. The submit tools record
    their result into a graph-state channel (so the checkpointer persists it);
    this before_model hook jumps to the agent's end node once that channel is set.
    """

    @before_model(can_jump_to=["end"])
    def _check_submitted(state: AgentState, runtime: Runtime):
        if state.get(channel):
            return {"jump_to": "end"}
        return None

    return [_check_submitted]


def build_model(
    model_id: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
):
    """Construct a chat model from env vars / explicit overrides.

    OpenRouter example:
        AUTODEBUG_MODEL=openrouter/owl-alpha
        AUTODEBUG_MODEL_PROVIDER=openai
        OPENAI_API_BASE=https://openrouter.ai/api/v1
    """
    mid = model_id or os.getenv("AUTODEBUG_MODEL", _DEFAULT_MODEL)
    prv = provider or os.getenv("AUTODEBUG_MODEL_PROVIDER", _DEFAULT_PROVIDER)
    kwargs: dict = {}
    resolved_base = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    if resolved_base and prv == "openai":
        kwargs["base_url"] = resolved_base
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(mid, model_provider=prv, **kwargs)
