"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import Budget, BudgetExceeded, build_model, budget_middleware
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage


def run_repro(state: DebugState, *, registry) -> DebugState:
    """Drive the repro agent until it produces a confirmed reproducing script."""
    assert state.repo_local_path
    cfg = registry.get_config("repro")
    repo_path = Path(state.repo_local_path)
    sandbox = Sandbox(str(repo_path))
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

    for attempt in range(cfg.max_retries + 1):
        budget = Budget.from_config(cfg)
        result.clear()
        tools = registry.build_tools(
            "repro",
            repo_path=repo_path,
            sandbox=sandbox,
            result=result,
        )
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=budget_middleware(budget),
        )

        try:
            agent.invoke(
                {"messages": [HumanMessage(content=initial_text)]},
                config={"recursion_limit": sys.maxsize},
            )
        except BudgetExceeded:
            state.total_tokens += budget.tokens_used
            state.total_cost += budget.cost_used
            if attempt < cfg.max_retries:
                continue
            state.stage = PipelineStage.FAILED
            state.error = f"ReproAgent: budget exceeded after {attempt + 1} attempt(s)"
            return state

        state.total_tokens += budget.tokens_used
        state.total_cost += budget.cost_used

        if result:
            state.repro = result[0]
            state.stage = PipelineStage.BISECT
            return state

    state.stage = PipelineStage.FAILED
    state.error = "ReproAgent: failed to reproduce the bug"
    return state
