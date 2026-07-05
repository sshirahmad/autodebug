"""Manager Agent — the "brain" that orchestrates the sub-agents via an FSM.

Unlike the linear pipeline (clone -> repro -> bisect -> root_cause -> fix), the
manager is itself a LangChain agent whose *tools are the sub-agents*. A finite
state machine (autodebug/fsm.py) governs which sub-agents it may call and which
system prompt it sees at each phase. Sub-agent results drive the transitions, so
the manager can run them in sequence and loop back (fix <-> repro/root_cause)
when a fix fails — letting the agents effectively communicate through it.

Activated by run_pipeline when config/agents/manager.json exists (and
AUTODEBUG_MANAGER != 0). The classic linear pipeline remains the fallback.
"""

from __future__ import annotations

import sys
import uuid

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

import os

from autodebug import resume
from autodebug.agent_state import ManagerAgentState
from autodebug.agents.base import (
    Budget, BudgetExceeded, build_model, budget_middleware, model_retry_middleware,
    planning_middleware, require_tool_calls_middleware, session_budget_middleware,
    submission_middleware, summarization_middleware, tool_call_limit_middleware,
)
from autodebug.fsm import (
    MANAGER_ALLOWED_TOOLS, ManagerPhase, fsm_prompt, fsm_tool_enforce, fsm_tool_gate,
    stage_hitl_middleware,
)
from autodebug.state import DebugState, PipelineStage

# Result/accounting fields the Manager subgraph owns — copied back out of the
# checkpointed `debug` channel into the caller's DebugState after the run.
_RESULT_FIELDS = (
    "repro", "bisect", "root_cause", "fix", "hypothesis_attempts",
    "total_tokens", "total_cost", "total_llm_calls", "stage", "error", "user_feedback",
)


def _progress(graph_state) -> str:
    """Compact, always-current status appended to every manager prompt, read from
    the checkpointed graph state (debug + fsm_phase channels)."""
    ds = DebugState(**((graph_state or {}).get("debug") or {"repo_url": "", "bug_report": ""}))
    phase = (graph_state or {}).get("fsm_phase") or ManagerPhase.INIT.value

    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    culprit = ds.bisect.culprit_commit[:12] if ds.bisect else "—"
    return (
        "## Progress\n"
        f"- phase: **{phase}**\n"
        f"- {mark(bool(ds.repro and ds.repro.confirmed))} reproduction confirmed\n"
        f"- {mark(bool(ds.bisect))} culprit commit: {culprit}\n"
        f"- {mark(bool(ds.root_cause))} root cause hypothesis\n"
        f"- {mark(bool(ds.fix))} fix verified"
    )


def manager_initial_text(bug_report: str, user_feedback: str | None = None) -> str:
    """The opening human message: the bug report + orchestration framing, plus any
    human-in-the-loop guidance from a prior failed attempt."""
    text = (
        f"Bug report:\n\n{bug_report}\n\n"
        "Orchestrate the sub-agents to reproduce, locate, explain, and fix this "
        "bug. The reproduction your repro agent builds is the success criterion — "
        "there is no provided test. Begin by reproducing it."
    )
    if user_feedback and user_feedback.strip():
        text += (
            "\n\nA previous attempt did NOT succeed. A developer reviewed it and "
            f"gave this guidance — follow it:\n\n{user_feedback.strip()}"
        )
    return text


def build_manager_agent(registry, *, checkpointer=None, hitl=False):
    """Compile the Manager `create_agent` graph + its session budget.

    With ``checkpointer=None`` the agent can be embedded as a **subgraph node** in a
    served graph — its `interrupt()`s then bubble to the parent stream (Studio /
    Agent Chat UI) and resume continues the same conversation. `run_manager` compiles
    it with the persistent `resume.get_saver()` for eval/CLI.

    ``hitl`` is the single switch for human-in-the-loop: when True it attaches BOTH
    triggers — stage-stuck (`stage_hitl_middleware`) and budget-exhausted (the
    interrupt mode of `session_budget_middleware`). Unattended/eval pass ``hitl=False``
    so a failure never blocks on input that won't come.
    """
    cfg = registry.get_config("manager")
    prompts = registry.prompt_states("manager")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)
    # Session-wide ceiling across the manager + every sub-agent it delegates to.
    # Per-agent budgets don't bound the REVISING loop, so cap the whole session.
    session_cost = float(os.getenv(
        "AUTODEBUG_SESSION_COST_BUDGET", cfg.extra.get("session_cost_budget_usd", 8.0)))
    session_secs = int(os.getenv(
        "AUTODEBUG_SESSION_TIME_BUDGET", cfg.extra.get("session_time_budget_seconds", 2400)))
    budget = Budget.from_config(cfg)
    tools = registry.build_tools("manager", registry=registry)
    agent = create_agent(
        model=llm,
        tools=tools,
        state_schema=ManagerAgentState,
        checkpointer=checkpointer,
        middleware=(
            budget_middleware(budget)
            + model_retry_middleware()
            + session_budget_middleware(session_cost, session_secs, hitl=hitl)
            + planning_middleware()
            + summarization_middleware(cfg.model, cfg.provider)
            + [fsm_prompt(prompts, progress_fn=_progress)]
            + [fsm_tool_gate(MANAGER_ALLOWED_TOOLS)]
            + [fsm_tool_enforce(MANAGER_ALLOWED_TOOLS)]
            + (stage_hitl_middleware() if hitl else [])
            + submission_middleware("outcome")
            + require_tool_calls_middleware()
            + tool_call_limit_middleware(cfg.tool_call_limits)
        ),
    )
    return agent, budget


def run_manager(state: DebugState, *, registry, hitl: bool = False) -> DebugState:
    """Drive the manager agent until it finishes (or the budget is exhausted).

    The pipeline state lives in the agent's checkpointed ``debug`` channel (seeded
    from `state`, read back at the end), not an external closure — so the same agent
    can be served as a resumable subgraph (see build_manager_agent). `hitl` is off by
    default (the eval/unattended path); the interactive graph builds the agent with
    `hitl=True` directly.
    """
    assert state.repo_volume, "manager requires a cloned repo volume"

    agent, budget = build_manager_agent(
        registry, checkpointer=resume.get_saver(), hitl=hitl)
    initial_text = manager_initial_text(state.bug_report, state.user_feedback)

    # Fresh thread per run: the manager REPLAYS its orchestration each time
    # (cheap — the sub-agents serve their results from cache on resume), rather
    # than resuming mid-conversation where a skipped sub-agent call would leave
    # the fixer without repro/bisect/root_cause.
    invoke_config = {
        "recursion_limit": sys.maxsize,
        "configurable": {"thread_id": f"manager-{resume.bug_key(state)}-{uuid.uuid4().hex[:8]}"},
    }
    initial_state = {
        "messages": [HumanMessage(content=initial_text)],
        "debug": state.model_dump(),
        "fsm_phase": ManagerPhase.INIT.value,
    }
    session_error = ""
    try:
        agent.invoke(initial_state, config=invoke_config)
    except BudgetExceeded as e:
        session_error = str(e)
    except Exception as e:  # noqa: BLE001 — never let one error abort the pipeline
        session_error = f"{type(e).__name__}: {str(e)[:300]}"

    # Read the pipeline state back out of the checkpointed `debug` channel into the
    # caller's DebugState (sub-agent results + their accumulated token/cost spend).
    final = resume.read_live(agent, invoke_config, "debug") or {}
    if final:
        updated = DebugState(**final)
        for f in _RESULT_FIELDS:
            setattr(state, f, getattr(updated, f))
    # The manager's OWN llm spend is on top of the sub-agents' (already in `debug`).
    state.total_tokens += budget.tokens_used
    state.total_cost += budget.cost_used
    state.total_llm_calls += budget.calls
    state.manager_phase = resume.read_live(agent, invoke_config, "fsm_phase")

    # Outcome: trust an explicit finish('success'); otherwise a verified fix on
    # state still counts. Everything else is a failure.
    outcome = resume.read_live(agent, invoke_config, "outcome")
    succeeded = bool(state.fix) and (
        not outcome or outcome.get("outcome") == "success"
    )
    if succeeded:
        state.stage = PipelineStage.DONE
        state.error = None
    else:
        state.stage = PipelineStage.FAILED
        if session_error:
            state.error = f"ManagerAgent: {session_error}"
        elif not state.error:
            reason = outcome["summary"] if outcome else "no verified fix produced"
            state.error = f"ManagerAgent: {reason}"
    return state
