"""Shared infrastructure: budget tracking, middleware, and model factory.

Each agent is built with LangChain's `create_agent` — see autodebug/agents/{repro,bisect,...}.py.
The agent's tool-execution loop is handled by create_agent; budgets are enforced
via the `budget_middleware` (a pair of before_model/after_model hooks).
"""

from __future__ import annotations

import contextvars
import logging
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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_PROVIDER = "anthropic"


# --- Token-level streaming -------------------------------------------------
# When enabled, every sub-agent's LLM tokens are forwarded to the active LangGraph
# stream writer as ``{"token": ...}`` custom events, so the interactive graph can
# surface them live (Studio / Agent Chat UI custom events, CLI inline). Gated so the
# default (and the eval harness) keep their clean, non-streaming behavior.
#
# Two switches: a process-wide env var (AUTODEBUG_STREAM_TOKENS=1) and a contextvar
# the graph sets per-run from its config (so a server can stream per-request without
# a global, race-free toggle). The contextvar is copied into the worker thread by
# ``asyncio.to_thread``, so a forwarder built inside a stage runner still sees it.
_STREAM_TOKENS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "autodebug_stream_tokens", default=False
)


def set_stream_tokens(on: bool) -> None:
    """Enable/disable token forwarding for the current execution context."""
    _STREAM_TOKENS.set(bool(on))


def _stream_tokens_active() -> bool:
    return _STREAM_TOKENS.get() or os.getenv("AUTODEBUG_STREAM_TOKENS", "0") == "1"


class _TokenStreamForwarder(BaseCallbackHandler):
    """Forwards each new LLM token to a LangGraph stream writer. Best-effort: a
    failed write (e.g. no live consumer) is swallowed so it never breaks the run."""

    def __init__(self, writer, agent: str | None = None):
        self._writer = writer
        self._agent = agent

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not token:
            return
        try:
            self._writer({"token": token, "agent": self._agent})
        except Exception:  # noqa: BLE001 — streaming is decorative, never fatal
            pass


def _token_forwarder(agent: str | None = None):
    """A forwarder callback if token streaming is on AND a stream writer is live in
    this context; otherwise None (so models are built unchanged)."""
    if not _stream_tokens_active():
        return None
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return None
    return _TokenStreamForwarder(writer, agent) if callable(writer) else None


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

    @property
    def fraction_used(self) -> float:
        """How close we are to the binding limit, as a fraction in [0, ∞).

        Max of the time fraction and the cost fraction — whichever ceiling the
        run is about to hit first. Used to nudge the agent to submit before the
        hard cap throws.
        """
        fractions = [0.0]
        if self.time_seconds:
            fractions.append(self.elapsed_seconds / self.time_seconds)
        if self.cost_usd:
            fractions.append(self.cost_used / self.cost_usd)
        return max(fractions)

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


def budget_nudge_middleware(budget: Budget, submit_tool: str, threshold: float = 0.8) -> list:
    """Salvage a result before the hard budget cap throws.

    Agents that investigate (root_cause) or iterate (fix) tend to spend their
    whole budget exploring and never call their submit tool — so they return
    *nothing*, the manager retries them, and the per-attempt cost multiplies
    until the session ceiling kills the run with no hypothesis or patch to show
    for it. This injects a one-time prompt once the agent crosses `threshold` of
    its time/cost budget, telling it to stop and submit its best current answer.
    A partial answer lets the pipeline proceed; an empty one wastes the budget.
    """
    fired = {"done": False}

    @before_model
    def _nudge(state: AgentState, runtime: Runtime) -> dict | None:
        if fired["done"] or budget.fraction_used < threshold:
            return None
        fired["done"] = True
        return {"messages": [HumanMessage(content=(
            f"⏳ You have used ~{budget.fraction_used:.0%} of your budget. STOP "
            f"investigating now and call `{submit_tool}` with your BEST current "
            "answer, based on the evidence you have already gathered. A partial, "
            "best-effort submission is far more useful than running out of budget "
            "with nothing submitted."
        ))]}

    return [_nudge]


def session_budget_middleware(max_cost: float | None, max_seconds: int | None,
                              hitl: bool = False) -> list:
    """Global ceiling across an orchestrated session (manager + all sub-agents).

    Per-agent budgets only bound a single sub-agent run; a manager that loops
    through REVISING can rack up many full sub-agent runs (black-1 hit ~$31). This
    before_model hook runs on each *manager* turn and inspects the cumulative spend
    in the ``debug`` graph-state channel (each sub-agent adds its cost there) plus
    wall-clock time.

    Unattended (``hitl=False``): raise BudgetExceeded once a ceiling is crossed — the
    run ends. Interactive (``hitl=True``): ``interrupt()`` with a summary instead; on
    resume the developer's reply is injected as guidance and another budget window is
    granted (``budget_extra`` += max_cost, clock reset) so the Manager keeps going —
    or ``skip`` ends it. This is the "budget exhausted" HITL trigger.
    """
    start = [time.monotonic()]  # mutable so a HITL resume can reset the clock

    @before_model
    def _check_session(graph_state: AgentState, runtime: Runtime):
        total_cost = float((graph_state.get("debug") or {}).get("total_cost", 0.0))
        extra = float(graph_state.get("budget_extra") or 0.0)
        elapsed = time.monotonic() - start[0]
        over_cost = max_cost is not None and total_cost > max_cost + extra
        over_time = max_seconds is not None and elapsed > max_seconds
        if not (over_cost or over_time):
            return None

        why = (f"cost ${total_cost:.2f} > ${max_cost + extra:.2f}" if over_cost
               else f"time {elapsed:.0f}s > {max_seconds}s")
        if not hitl:
            raise BudgetExceeded(f"Session budget exceeded ({why})")

        from langgraph.types import interrupt
        from langchain_core.messages import HumanMessage

        recap = ""
        ds = (graph_state.get("debug") or {})
        if ds.get("fix"):
            recap = "a fix was produced but not verified; "
        feedback = interrupt({
            "type": "budget_exhausted",
            "summary": (f"Session budget reached ({why}). {recap}Reply with guidance to "
                        "continue with a fresh budget window, or 'skip' to stop here."),
        })
        fb = str(feedback or "").strip()
        if fb.lower() == "skip":
            raise BudgetExceeded(f"Session budget exceeded ({why}); developer chose to stop")
        start[0] = time.monotonic()  # reset the time window
        update = {"budget_extra": extra + (max_cost or 0.0)}
        if fb:
            update["messages"] = [HumanMessage(content=f"[developer guidance — follow this]: {fb}")]
        return update

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


# Max consecutive text-only (no tool call) model responses before we stop nudging
# and end the run — bounds the retry when the model can't/won't call a tool.
_MAX_NO_TOOL_RETRIES = 5


def require_tool_calls_middleware() -> list:
    """Reject text-only AI responses and force the model to retry with a tool call.

    Modern reasoning models will happily emit thousands of tokens of analysis
    text without ever calling a tool — burning budget without making progress.
    After every model call, if the response has no tool_calls, we append a
    short corrective message and jump back to the model node so it tries again —
    up to _MAX_NO_TOOL_RETRIES times, then end the run (the jump bypasses the
    before_model budget check, so an unbounded loop can't be stopped by it).
    """
    from langchain_core.messages import AIMessage, HumanMessage

    nudge = ("Your last response had no tool call. Every response MUST "
             "be a tool call. Make a tool call now — do not write text.")

    @after_model(can_jump_to=["model", "end"])
    def _require_tool(state: AgentState, runtime: Runtime):
        msg = state["messages"][-1]
        if not isinstance(msg, AIMessage):
            return None
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            return None
        # Bound the retry. `jump_to: model` re-enters the model node DIRECTLY,
        # bypassing the before_model hooks — so the session budget check never runs
        # and can't stop us. If the model is broken (e.g. a pulled model 404ing on
        # every call, then degrading to a text error AIMessage via
        # ModelRetryMiddleware's on_failure='continue'), this would loop until the
        # graph recursion limit (a 2.5-hour, 0-cost GraphRecursionError). Count the
        # consecutive nudges already in the trajectory and give up after a few:
        # end the run cleanly (FAILED) instead of spinning.
        streak = 0
        for m in reversed(state["messages"][:-1]):
            if isinstance(m, HumanMessage) and str(getattr(m, "content", "")) == nudge:
                streak += 1
            elif isinstance(m, AIMessage):
                continue
            else:
                break
        if streak >= _MAX_NO_TOOL_RETRIES:
            return {"jump_to": "end"}
        return {"messages": [HumanMessage(content=nudge)], "jump_to": "model"}

    return [_require_tool]


# Cap how much trajectory we feed the optimizer — long fix runs can be 100+
# messages and the optimizer sends them all to an LLM. Keep the opening (task
# framing) plus the most recent turns, where the failure actually shows up.
_TRAJ_HEAD = 4
_TRAJ_TAIL = 56

# Prompt optimization runs on a DEDICATED model, separate from the agent's: the
# gradient optimizer makes a structured-output (tool_choice) call that some agent
# models — notably ones routed via OpenRouter — reject with a 404. Configure via
# AUTODEBUG_PROMPT_OPTIM_MODEL / _PROVIDER; this is the default and must support
# tool_choice.
_DEFAULT_OPTIM_MODEL = "openai/gpt-4o-mini"


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


# LangMem's gradient optimizer treats every `{x}` in the prompt as a required
# f-string variable (it `.format()`s the prompt + trajectory internally). Our
# system prompts are STATIC text — never formatted — but they contain literal
# braces (e.g. code examples in skills, `{sha}`, `{len(commits)}`), and so do
# agent trajectories. Those make the optimizer demand spurious "variables"
# ("Missing required variable: …") and blow up its own `.format()` (IndexError).
# So we mask braces to private-use sentinels for the round-trip and restore them.
_BRACE_L, _BRACE_R = chr(0xE000), chr(0xE001)  # Unicode PUA; never appear in prompts


def _mask_braces(s: str) -> str:
    return s.replace("{", _BRACE_L).replace("}", _BRACE_R)


def _unmask_braces(s: str) -> str:
    return s.replace(_BRACE_L, "{").replace(_BRACE_R, "}")


def _mask_messages(messages: list) -> list:
    """Copies of `messages` with braces in their text content masked."""
    out = []
    for m in messages:
        c = getattr(m, "content", None)
        if isinstance(c, str) and ("{" in c or "}" in c):
            try:
                m = m.model_copy(update={"content": _mask_braces(c)})
            except Exception:
                continue  # skip a message we can't safely copy rather than crash
        out.append(m)
    return out


def maybe_optimize_prompt(
    current_prompt: str,
    messages: list,
    feedback: str,
    *,
    model_id: str | None = None,
    provider: str | None = None,
) -> str:
    """Improve `current_prompt` from a failed attempt's trajectory via LangMem.

    The gradient optimizer makes a structured-output (tool_choice) call that some
    agent models can't do (OpenRouter routes may 404), so optimization runs on a
    DEDICATED model — `AUTODEBUG_PROMPT_OPTIM_MODEL` / `_PROVIDER` (default
    openai/gpt-4o-mini), which must support tool_choice — independent of the
    agent's own `model_id`. The provider falls back to the agent's so the existing
    keys/routing are reused. Best-effort: any error returns the original prompt so
    the retry loop is never blocked. Disable with AUTODEBUG_PROMPT_OPTIM=0.
    """
    if os.getenv("AUTODEBUG_PROMPT_OPTIM", "1") == "0" or not messages:
        return current_prompt
    opt_model = os.getenv("AUTODEBUG_PROMPT_OPTIM_MODEL", _DEFAULT_OPTIM_MODEL)
    opt_provider = os.getenv("AUTODEBUG_PROMPT_OPTIM_PROVIDER") or provider
    try:
        from langmem import create_prompt_optimizer
        from langmem.prompts.types import AnnotatedTrajectory

        optimizer = create_prompt_optimizer(
            build_model(model_id=opt_model, provider=opt_provider), kind="gradient"
        )
        # Mask braces so LangMem doesn't read literal {…} in the prompt/trajectory
        # as required f-string variables (or crash its internal .format()).
        trajectory = AnnotatedTrajectory(
            messages=_mask_messages(_trim_trajectory(messages)),
            feedback=_mask_braces(feedback),
        )
        improved = optimizer.invoke(
            {"trajectories": [trajectory], "prompt": _mask_braces(current_prompt)}
        )
        if isinstance(improved, str) and improved.strip():
            return _unmask_braces(improved)
    except Exception as exc:
        logger.warning(
            "Prompt optimization failed on model %r (provider %r); keeping the "
            "original prompt. %s: %s",
            opt_model, opt_provider, type(exc).__name__, exc,
        )
    return current_prompt


# ---------------------------------------------------------------------------
# Adversarial audit (LLM-as-judge)
# ---------------------------------------------------------------------------
# A cheap, dedicated model reviews an agent's submission before it's accepted and
# flags real defects the agent's own checks miss — a repro too weak to discriminate
# a correct fix, or a fix that masks the symptom / over-reaches. It runs on a plain
# text verdict (no tool_choice), so any model works. Disable with AUTODEBUG_JUDGE=0;
# pick the model with AUTODEBUG_JUDGE_MODEL / _PROVIDER.
_DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

_JUDGE_PROMPTS = {
    "repro": (
        "You are an ADVERSARIAL reviewer of a bug REPRODUCTION script. This script "
        "is the ONLY oracle the fixer optimizes against, so a weak one lets a wrong "
        "or over-eager fix pass. Reply REVISE if the repro: passes while the bug is "
        "still unfixed; would STILL fail after a correct fix (tests the wrong thing); "
        "only checks the symptom instead of the full expected behavior; or is overfit "
        "to a single input so a special-cased fix would satisfy it. Reply ACCEPT only "
        "if it clearly distinguishes a correct fix from both the buggy code and a "
        "plausible over-eager fix. Be strict but fair — do not invent flaws."
    ),
    "fix": (
        "You are an ADVERSARIAL reviewer of a code FIX for the stated root cause. "
        "Reply REVISE if the patch: likely doesn't address the actual cause (masks "
        "the symptom); is incomplete; or is over-broad and could change unrelated "
        "behavior. Reply ACCEPT if it is a minimal, targeted change consistent with "
        "the root cause. Be strict but fair — do not invent flaws."
    ),
}


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Parse a judge reply into (accept, critique). Defaults to accept unless the
    reply clearly says REVISE — a malformed/empty reply never blocks the pipeline."""
    upper = text.upper()
    revise = "REVISE" in upper and "VERDICT: ACCEPT" not in upper and "VERDICT:ACCEPT" not in upper
    if not revise:
        return True, ""
    idx = upper.find("CRITIQUE:")
    critique = text[idx + len("CRITIQUE:"):].strip() if idx != -1 else text.strip()
    return False, critique[:600]


def maybe_audit(
    kind: str, payload: str, *, model_id: str | None = None, provider: str | None = None
) -> tuple[bool, str]:
    """Adversarially review a submission with a cheap dedicated judge.

    Returns (ok, critique): ok=True means accept; ok=False means there's a real
    flaw and `critique` says what to fix. Best-effort — any error, an unknown
    `kind`, or AUTODEBUG_JUDGE=0 returns (True, "") so the pipeline is never
    blocked by the judge.
    """
    if os.getenv("AUTODEBUG_JUDGE", "1") == "0" or kind not in _JUDGE_PROMPTS:
        return True, ""
    jmodel = os.getenv("AUTODEBUG_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
    jprovider = os.getenv("AUTODEBUG_JUDGE_PROVIDER") or provider
    try:
        from langchain_core.messages import SystemMessage

        model = build_model(model_id=jmodel, provider=jprovider)
        instruction = (
            "\n\nAnswer in exactly this form:\nVERDICT: ACCEPT or REVISE\n"
            "CRITIQUE: <one or two concrete, actionable sentences>"
        )
        resp = model.invoke([
            SystemMessage(content=_JUDGE_PROMPTS[kind] + instruction),
            HumanMessage(content=payload[:6000]),
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _parse_verdict(content)
    except Exception as exc:
        logger.warning(
            "Audit (%s) failed on model %r (provider %r); accepting. %s: %s",
            kind, jmodel, jprovider, type(exc).__name__, exc,
        )
        return True, ""


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
    # Per-request timeout so a stalled provider connection RAISES instead of
    # blocking on a C-level socket read forever (which also makes Ctrl+C
    # undeliverable). model_retry_middleware then retries the timeout, and the
    # budget eventually trips. Set AUTODEBUG_MODEL_TIMEOUT=0 to disable.
    timeout = float(os.getenv("AUTODEBUG_MODEL_TIMEOUT", "120"))
    if timeout > 0:
        kwargs["timeout"] = timeout
    # Token streaming: with streaming=True, even .invoke() streams the response
    # internally and fires on_llm_new_token, which our forwarder relays to the
    # graph stream. No-op (model built unchanged) unless streaming is enabled and a
    # writer is live.
    forwarder = _token_forwarder()
    if forwarder is not None:
        kwargs["streaming"] = True
    model = init_chat_model(mid, model_provider=prv, **kwargs)
    if forwarder is not None:
        model = model.with_config({"callbacks": [forwarder]})
    return model


def model_for_attempt(attempt: int, model_id: str | None, provider: str | None):
    """Build the chat model for a retry `attempt`.

    On retries (attempt > 0), escalate to AUTODEBUG_RETRY_MODEL / _PROVIDER if
    configured — the cheap base model takes the first shot, a stronger model takes
    the do-overs. Falls back to the base model when no retry model is set, so this
    is a no-op unless you opt in.
    """
    if attempt > 0:
        retry_model = os.getenv("AUTODEBUG_RETRY_MODEL")
        if retry_model:
            return build_model(
                model_id=retry_model,
                provider=os.getenv("AUTODEBUG_RETRY_MODEL_PROVIDER") or provider,
            )
    return build_model(model_id=model_id, provider=provider)
