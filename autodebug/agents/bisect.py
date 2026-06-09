"""Bisect Agent — finds the commit that introduced a regression."""

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
from autodebug.state import BisectResult, DebugState, PipelineStage
from autodebug.tools import git_utils


def run_bisect(state: DebugState, *, registry) -> DebugState:
    """Drive the bisect agent until it submits a culprit commit SHA."""
    assert state.repo_volume
    assert state.repro and state.repro.confirmed

    cfg = registry.get_config("bisect")
    result: list[BisectResult] = []
    known_good = state.known_good_commit or ""

    system_prompt = registry.system_prompt("bisect")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    with Sandbox(volume=state.repo_volume) as sandbox:
        git_utils.unshallow(sandbox)
        original_sha = git_utils.current_sha(sandbox)
        head_info = git_utils.get_commit_info(sandbox, original_sha)

        initial_text = (
            f"Bug report:\n{state.bug_report}\n\n"
            f"Current HEAD (last buggy state): {head_info.short_sha} — {head_info.message}\n"
            f"Date: {head_info.date}\n"
            + (f"Known-good commit (bug did NOT exist here): {known_good}\n" if known_good else "")
            + f"\nRepro script (fails at HEAD, exits non-zero when bug is present):\n"
            f"```python\n{state.repro.repro_script}\n```\n\n"
            f"Repro error output:\n```\n{state.repro.error_output[:2000]}\n```\n\n"
            "Find the FIRST commit between the known-good commit and HEAD that introduced "
            "this bug, then call `submit_culprit` with its SHA."
        )

        crashed = False
        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools("bisect", sandbox=sandbox, result=result)
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
                "configurable": {"thread_id": f"bisect-{attempt}"},
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

            # Accept a submitted result even if the attempt later crashed —
            # the agent often submits then keeps over-investigating until the
            # budget hits, and we'd otherwise throw away a valid answer.
            if result:
                state.bisect = result[0]
                state.stage = PipelineStage.ROOT_CAUSE
                return state

            # No result. Retry only if the attempt crashed AND retries remain.
            if not crashed or attempt >= cfg.max_retries:
                break

            # Retrying: optimize the prompt from this failed attempt's history.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("identifying the culprit commit and calling submit_culprit"),
                model_id=cfg.model, provider=cfg.provider,
            )

        state.stage = PipelineStage.FAILED
        state.error = (
            "BisectAgent: budget exceeded with no submission"
            if crashed else "BisectAgent: could not identify culprit commit"
        )
        return state
