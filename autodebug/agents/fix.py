"""Fix Agent — generates and validates patches."""

from __future__ import annotations

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import Budget, BudgetExceeded, build_model, budget_middleware
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage


def run_fix(state: DebugState, *, registry) -> DebugState:
    """Drive the fix agent until it submits a passing patch."""
    assert state.repo_local_path
    assert state.repro
    assert state.bisect
    assert state.root_cause

    cfg = registry.get_config("fix")
    repo_path = Path(state.repo_local_path)
    sandbox = Sandbox(str(repo_path))
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
        + f"\n\nCulprit commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n"
        + test_cmd_note
        + "\nPlease fix this bug. Start by reading the relevant files."
    )

    # Fix agent may use a stronger model than the rest (configurable)
    model_id = cfg.model or os.getenv("AUTODEBUG_FIX_MODEL")
    provider = cfg.provider or os.getenv("AUTODEBUG_FIX_MODEL_PROVIDER")
    system_prompt = registry.system_prompt("fix")
    llm = build_model(model_id=model_id, provider=provider)

    for attempt in range(cfg.max_retries + 1):
        budget = Budget.from_config(cfg)
        result.clear()
        patches.clear()
        tools = registry.build_tools(
            "fix",
            repo_path=repo_path,
            sandbox=sandbox,
            repro_script=state.repro.repro_script,
            test_command=state.test_command or "",
            patches=patches,
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
                config={"recursion_limit": 100},
            )
        except BudgetExceeded:
            state.total_tokens += budget.tokens_used
            state.total_cost += budget.cost_used
            if attempt < cfg.max_retries:
                continue
            state.stage = PipelineStage.FAILED
            state.error = f"FixAgent: budget exceeded after {attempt + 1} attempt(s)"
            return state

        state.total_tokens += budget.tokens_used
        state.total_cost += budget.cost_used

        if result:
            state.fix = result[0]
            state.stage = PipelineStage.DONE
            return state

    state.stage = PipelineStage.FAILED
    state.error = "FixAgent: could not produce a valid fix"
    return state
