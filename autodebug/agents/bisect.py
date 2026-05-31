"""Bisect Agent — finds the commit that introduced a regression."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import Budget, BudgetExceeded, build_model, budget_middleware
from autodebug.sandbox import Sandbox
from autodebug.state import BisectResult, DebugState, PipelineStage
from autodebug.tools import git_utils


def run_bisect(state: DebugState, *, registry) -> DebugState:
    """Drive the bisect agent until it submits a culprit commit SHA."""
    assert state.repo_local_path
    assert state.repro and state.repro.confirmed

    cfg = registry.get_config("bisect")
    repo = str(state.repo_local_path)
    sandbox = Sandbox(repo)

    git_utils.unshallow(repo)
    original_sha = git_utils.current_sha(repo)
    head_info = git_utils.get_commit_info(repo, original_sha)

    result: list[BisectResult] = []
    known_good = state.known_good_commit or ""

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

    system_prompt = registry.system_prompt("bisect")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    for attempt in range(cfg.max_retries + 1):
        budget = Budget.from_config(cfg)
        result.clear()
        tools = registry.build_tools(
            "bisect",
            repo_path=Path(repo),
            sandbox=sandbox,
            repo=repo,
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
            state.error = f"BisectAgent: budget exceeded after {attempt + 1} attempt(s)"
            return state

        state.total_tokens += budget.tokens_used
        state.total_cost += budget.cost_used

        if result:
            state.bisect = result[0]
            state.stage = PipelineStage.ROOT_CAUSE
            return state

    state.stage = PipelineStage.FAILED
    state.error = "BisectAgent: could not identify culprit commit"
    return state
