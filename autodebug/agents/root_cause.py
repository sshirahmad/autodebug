"""Root Cause Agent — explains WHY the culprit commit broke things."""

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


def run_root_cause(state: DebugState, *, registry) -> DebugState:
    """Drive the root_cause agent until it submits a hypothesis."""
    assert state.repo_volume
    assert state.bisect
    assert state.repro

    cfg = registry.get_config("root_cause")
    result: list = []

    system_prompt = registry.system_prompt("root_cause")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    with Sandbox(volume=state.repo_volume) as sandbox:
        initial_run = sandbox.run_script(state.repro.repro_script)
        initial_text = (
            f"Culprit commit: {state.bisect.commit_message}\n"
            f"SHA: {state.bisect.culprit_commit}\n\n"
            f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
            f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n\n"
            "Please identify the root cause of this regression."
        )

        crashed = False
        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools(
                "root_cause",
                sandbox=sandbox,
                parent_sha=f"{state.bisect.culprit_commit}^",
                repro_script=state.repro.repro_script,
                result=result,
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
                "configurable": {"thread_id": f"root_cause-{attempt}"},
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

            if result:
                state.root_cause = result[0]
                state.stage = PipelineStage.FIX
                return state

            if not crashed or attempt >= cfg.max_retries:
                break

            # Retrying: optimize the prompt from this failed attempt's history.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("determining the root cause and calling submit_root_cause"),
                model_id=cfg.model, provider=cfg.provider,
            )

    state.stage = PipelineStage.FAILED
    state.error = (
        "RootCauseAgent: budget exceeded with no submission"
        if crashed else "RootCauseAgent: could not determine root cause"
    )
    return state
