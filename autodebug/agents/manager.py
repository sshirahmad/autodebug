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
    FSM, MANAGER_ALLOWED_TOOLS, fsm_prompt, fsm_tool_enforce, fsm_tool_gate,
)
from autodebug.state import DebugState, PipelineStage


def _progress_block(state: DebugState, fsm: FSM) -> str:
    """Compact, always-current status appended to every manager prompt."""
    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    culprit = state.bisect.culprit_commit[:12] if state.bisect else "—"
    return (
        "## Progress\n"
        f"- phase: **{fsm.phase}**\n"
        f"- {mark(bool(state.repro and state.repro.confirmed))} reproduction confirmed\n"
        f"- {mark(bool(state.bisect))} culprit commit: {culprit}\n"
        f"- {mark(bool(state.root_cause))} root cause hypothesis\n"
        f"- {mark(bool(state.fix))} fix verified"
    )


def run_manager(state: DebugState, *, registry) -> DebugState:
    """Drive the manager agent until it finishes (or the budget is exhausted)."""
    assert state.repo_volume, "manager requires a cloned repo volume"

    cfg = registry.get_config("manager")
    fsm = FSM()
    prompts = registry.prompt_states("manager")

    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    initial_text = (
        f"Bug report:\n\n{state.bug_report}\n\n"
        "Orchestrate the sub-agents to reproduce, locate, explain, and fix this "
        "bug. The reproduction your repro agent builds is the success criterion — "
        "there is no provided test. Begin by reproducing it."
    )
    # A previous attempt failed and a developer stepped in (human-in-the-loop);
    # surface their guidance so this attempt is steered rather than a blind replay.
    if state.user_feedback and state.user_feedback.strip():
        initial_text += (
            "\n\nA previous attempt did NOT succeed. A developer reviewed it and "
            f"gave this guidance — follow it:\n\n{state.user_feedback.strip()}"
        )

    # Session-wide ceiling across the manager + every sub-agent it delegates to.
    # Per-agent budgets don't bound the REVISING loop, so cap the whole session.
    session_cost = float(os.getenv(
        "AUTODEBUG_SESSION_COST_BUDGET",
        cfg.extra.get("session_cost_budget_usd", 8.0),
    ))
    session_secs = int(os.getenv(
        "AUTODEBUG_SESSION_TIME_BUDGET",
        cfg.extra.get("session_time_budget_seconds", 2400),
    ))

    budget = Budget.from_config(cfg)
    tools = registry.build_tools(
        "manager", state=state, registry=registry, fsm=fsm,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        state_schema=ManagerAgentState,
        checkpointer=resume.get_saver(),
        middleware=(
            budget_middleware(budget)
            + model_retry_middleware()
            + session_budget_middleware(state, session_cost, session_secs)
            + planning_middleware()
            + summarization_middleware(cfg.model, cfg.provider)
            + [fsm_prompt(fsm, prompts, progress_fn=lambda: _progress_block(state, fsm))]
            + [fsm_tool_gate(fsm, MANAGER_ALLOWED_TOOLS)]
            + [fsm_tool_enforce(fsm, MANAGER_ALLOWED_TOOLS)]
            + submission_middleware("outcome")
            + require_tool_calls_middleware()
            + tool_call_limit_middleware(cfg.tool_call_limits)
        ),
    )

    # Fresh thread per run: the manager REPLAYS its orchestration each time
    # (cheap — the sub-agents serve their results from cache on resume), rather
    # than resuming mid-conversation where a skipped sub-agent call would leave
    # state.repro/bisect/root_cause empty for the fixer.
    invoke_config = {
        "recursion_limit": sys.maxsize,
        "configurable": {"thread_id": f"manager-{resume.bug_key(state)}-{uuid.uuid4().hex[:8]}"},
    }
    session_error = ""
    try:
        agent.invoke({"messages": [HumanMessage(content=initial_text)]}, config=invoke_config)
    except BudgetExceeded as e:
        session_error = str(e)
    except Exception as e:  # noqa: BLE001 — never let one error abort the pipeline
        session_error = f"{type(e).__name__}: {str(e)[:300]}"

    state.total_tokens += budget.tokens_used
    state.total_cost += budget.cost_used
    state.total_llm_calls += budget.calls
    state.manager_phase = str(fsm.phase)

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
