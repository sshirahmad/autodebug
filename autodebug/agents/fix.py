"""Fix Agent — generates and validates patches."""

from __future__ import annotations

import os
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


def run_fix(state: DebugState, *, registry) -> DebugState:
    """Drive the fix agent until it submits a passing patch."""
    assert state.repo_volume
    assert state.repro
    assert state.bisect
    assert state.root_cause

    cfg = registry.get_config("fix")
    patches: list[dict] = []
    result: list = []

    test_cmd_note = (
        f"\nTest command to verify fix: `{state.test_command}`\n"
        if state.test_command else ""
    )
    initial_text = (
        f"Root cause:\n{state.root_cause.hypothesis}\n\n"
        f"{state.root_cause.summary}\n\n"
        "Relevant lines:\n"
        + "\n".join(f"  - {l}" for l in state.root_cause.relevant_lines)
        + f"\n\nReproduction script (the success criterion — this MUST exit 0 "
          f"after your fix; read it carefully to see what the fix must produce):\n"
          f"```python\n{state.repro.repro_script}\n```\n"
        + f"\nCulprit commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n"
        + test_cmd_note
        + "\nPlease fix this bug."
    )

    model_id = cfg.model or os.getenv("AUTODEBUG_FIX_MODEL")
    provider = cfg.provider or os.getenv("AUTODEBUG_FIX_MODEL_PROVIDER")
    system_prompt = registry.system_prompt("fix")
    llm = build_model(model_id=model_id, provider=provider)

    crashed = False
    with Sandbox(volume=state.repo_volume) as sandbox:
        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools(
                "fix",
                sandbox=sandbox,
                repro_script=state.repro.repro_script,
                test_command=state.test_command or "",
                patches=patches,
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
                    + summarization_middleware(model_id, provider)
                    + submission_middleware(result)
                    + require_tool_calls_middleware()
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": f"fix-{attempt}"},
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
                state.fix = result[0]
                state.stage = PipelineStage.DONE
                return state

            if not crashed or attempt >= cfg.max_retries:
                break

            # Retrying: optimize the prompt from this failed attempt's history.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("producing a patch that passes the repro and tests, then calling submit_fix"),
                model_id=model_id, provider=provider,
            )

    state.stage = PipelineStage.FAILED
    state.error = (
        "FixAgent: budget exceeded with no submission"
        if crashed else "FixAgent: could not produce a valid fix"
    )
    return state
