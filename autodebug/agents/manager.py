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

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import (
    Budget, BudgetExceeded, build_model, budget_middleware, planning_middleware,
    require_tool_calls_middleware, submission_middleware, summarization_middleware,
    tool_call_limit_middleware,
)
from autodebug.fsm import (
    FSM, MANAGER_ALLOWED_TOOLS, fsm_prompt, fsm_tool_gate,
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
    result: list = []
    prompts = registry.prompt_states("manager")

    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    test_hint = (
        f"\nKnown failing test (the fix must make this pass): `{state.test_command}`\n"
        if state.test_command else ""
    )
    initial_text = (
        f"Bug report:\n\n{state.bug_report}\n"
        f"{test_hint}\n"
        "Orchestrate the sub-agents to reproduce, locate, explain, and fix this "
        "bug. Begin by reproducing it."
    )

    budget = Budget.from_config(cfg)
    tools = registry.build_tools(
        "manager", state=state, registry=registry, fsm=fsm, result=result,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=(
            budget_middleware(budget)
            + planning_middleware()
            + summarization_middleware(cfg.model, cfg.provider)
            + [fsm_prompt(fsm, prompts, progress_fn=lambda: _progress_block(state, fsm))]
            + [fsm_tool_gate(fsm, MANAGER_ALLOWED_TOOLS)]
            + submission_middleware(result)
            + require_tool_calls_middleware()
            + tool_call_limit_middleware(cfg.tool_call_limits)
        ),
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage(content=initial_text)]},
            config={"recursion_limit": sys.maxsize},
        )
    except BudgetExceeded:
        pass

    state.total_tokens += budget.tokens_used
    state.total_cost += budget.cost_used
    state.manager_phase = str(fsm.phase)

    # Outcome: trust an explicit finish('success'); otherwise a verified fix on
    # state still counts. Everything else is a failure.
    succeeded = bool(state.fix) and (
        not result or result[0].get("outcome") == "success"
    )
    if succeeded:
        state.stage = PipelineStage.DONE
        state.error = None
    else:
        state.stage = PipelineStage.FAILED
        if not state.error:
            reason = result[0]["summary"] if result else "no verified fix produced"
            state.error = f"ManagerAgent: {reason}"
    return state
