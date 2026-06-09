"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from autodebug.agents.base import (
    Budget, BudgetExceeded, attempt_trajectory, build_model, budget_middleware,
    maybe_optimize_prompt, planning_middleware, require_tool_calls_middleware,
    retry_feedback, submission_middleware, summarization_middleware,
    tool_call_limit_middleware,
)
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage


def run_repro(state: DebugState, *, registry) -> DebugState:
    """Drive the repro agent until it produces a confirmed reproducing script."""
    assert state.repo_volume
    cfg = registry.get_config("repro")
    result: list = []

    test_cmd_hint = (
        f"\nKnown failing test (run this first to see the failure):\n"
        f"  `{state.test_command}`\n"
        if state.test_command else ""
    )
    initial_text = (
        f"Bug report:\n\n{state.bug_report}\n"
        f"{test_cmd_hint}\n"
        + (
            "Start by running the failing test above to observe the error output, "
            "then write a minimal Python script that reproduces the same failure.\n"
            if state.test_command else
            "Please reproduce this bug.\n"
        )
    )

    system_prompt = registry.system_prompt("repro")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    crashed = False
    with Sandbox(volume=state.repo_volume) as sandbox:
        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools(
                "repro", sandbox=sandbox, result=result,
                test_command=state.test_command or "",
            )
            saver = InMemorySaver()
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                checkpointer=saver,
                middleware=(
                    budget_middleware(budget)
                    + planning_middleware()
                    + summarization_middleware(cfg.model, cfg.provider)
                    + submission_middleware(result)
                    + require_tool_calls_middleware()
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": f"repro-{attempt}"},
            }

            crashed = False
            try:
                agent.invoke(
                    {"messages": [HumanMessage(content=initial_text)]},
                    config=invoke_config,
                )
            except BudgetExceeded:
                crashed = True

            state.total_tokens += budget.tokens_used
            state.total_cost += budget.cost_used

            # Accept a submission even if the attempt later crashed.
            if result:
                state.repro = result[0]
                state.stage = PipelineStage.BISECT
                return state

            if not crashed or attempt >= cfg.max_retries:
                break

            # Retrying: optimize the prompt from this failed attempt's history.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("reproducing the bug and calling submit_repro"),
                model_id=cfg.model, provider=cfg.provider,
            )

    state.stage = PipelineStage.FAILED
    state.error = (
        "ReproAgent: budget exceeded with no submission"
        if crashed else "ReproAgent: failed to reproduce the bug"
    )
    return state
