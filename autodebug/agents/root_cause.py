"""Root Cause Agent — explains WHY the culprit commit broke things."""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import Budget, BudgetExceeded, build_model, budget_middleware
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage


def run_root_cause(state: DebugState, *, registry) -> DebugState:
    """Drive the root_cause agent until it submits a hypothesis."""
    assert state.repo_local_path
    assert state.bisect
    assert state.repro

    cfg = registry.get_config("root_cause")
    repo_path = Path(state.repo_local_path)
    sandbox = Sandbox(str(repo_path))
    result: list = []

    initial_run = sandbox.run_script(state.repro.repro_script)
    initial_text = (
        f"Culprit commit: {state.bisect.commit_message}\n"
        f"SHA: {state.bisect.culprit_commit}\n\n"
        f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
        f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n\n"
        "Please identify the root cause of this regression."
    )

    system_prompt = registry.system_prompt("root_cause")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    for attempt in range(cfg.max_retries + 1):
        budget = Budget.from_config(cfg)
        result.clear()
        tools = registry.build_tools(
            "root_cause",
            repo_path=repo_path,
            sandbox=sandbox,
            parent_sha=f"{state.bisect.culprit_commit}^",
            repro_script=state.repro.repro_script,
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
            state.error = f"RootCauseAgent: budget exceeded after {attempt + 1} attempt(s)"
            return state

        state.total_tokens += budget.tokens_used
        state.total_cost += budget.cost_used

        if result:
            state.root_cause = result[0]
            state.stage = PipelineStage.FIX
            return state

    state.stage = PipelineStage.FAILED
    state.error = "RootCauseAgent: could not determine root cause"
    return state
