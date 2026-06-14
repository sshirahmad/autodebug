"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import (
    Budget, BudgetExceeded, attempt_trajectory, build_model, budget_middleware,
    maybe_optimize_prompt, model_retry_middleware, planning_middleware,
    require_tool_calls_middleware, retry_feedback, submission_middleware,
    summarization_middleware, tool_call_limit_middleware,
)
from autodebug import resume
from autodebug.agent_state import ReproAgentState
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, ReproResult


def run_repro(state: DebugState, *, registry) -> DebugState:
    """Drive the repro agent until it produces a confirmed reproducing script."""
    assert state.repo_volume
    cfg = registry.get_config("repro")

    # Refine rather than regenerate: if a reproduction already exists, we're being
    # re-invoked (the manager looped back from a failed fix). Hand the agent the
    # prior script so it corrects it instead of inventing a different one each time.
    refine_note = ""
    if state.repro is not None:
        refine_note = (
            "\nNOTE: a previous reproduction was confirmed but the downstream fix "
            "could not be verified against it — it may be imprecise or target the "
            "wrong failure. Previous reproduction:\n"
            f"```python\n{state.repro.repro_script}\n```\n"
            f"Previous observed failure:\n{state.repro.error_output[:1000]}\n"
            "Refine THIS reproduction so it precisely targets the bug described "
            "above; do not start from an unrelated angle.\n"
        )
    initial_text = (
        f"Bug report:\n\n{state.bug_report}\n\n"
        f"{refine_note}"
        "Reproduce this bug: explore the repository, then write a minimal Python "
        "script that triggers the failure described above and confirm it fails.\n"
    )

    system_prompt = registry.system_prompt("repro")
    llm = build_model(model_id=cfg.model, provider=cfg.provider)

    crashed = False
    run_error: str | None = None
    key = resume.bug_key(state)
    saver = resume.get_saver()
    with Sandbox(volume=state.repo_volume) as sandbox:
        # Resume: a prior run already produced (and we re-validate) a repro.
        cached = resume.cached_repro(key, cfg.max_retries, sandbox)
        if cached is not None:
            state.repro = cached
            state.stage = PipelineStage.BISECT
            return state
        resume.clear("repro", key, cfg.max_retries)  # drop stale threads before a fresh run

        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools("repro", sandbox=sandbox)
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                state_schema=ReproAgentState,
                checkpointer=saver,
                middleware=(
                    budget_middleware(budget)
                    + model_retry_middleware()
                    + planning_middleware()
                    + summarization_middleware(cfg.model, cfg.provider)
                    + submission_middleware("repro")
                    + require_tool_calls_middleware()
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": resume.thread_id("repro", key, attempt)},
            }

            crashed = False
            try:
                agent.invoke(
                    {"messages": [HumanMessage(content=initial_text)]},
                    config=invoke_config,
                )
            except BudgetExceeded:
                crashed = True
            except Exception as exc:  # noqa: BLE001 — degrade, don't abort the pipeline
                crashed = True
                run_error = f"{type(exc).__name__}: {str(exc)[:300]}"

            state.total_tokens += budget.tokens_used
            state.total_cost += budget.cost_used
            state.total_llm_calls += budget.calls

            # Accept a submission even if the attempt later crashed — the submit
            # tool persists it to the `repro` channel before the budget trips.
            submitted = resume.read_live(agent, invoke_config, "repro")
            if submitted:
                state.repro = ReproResult(**submitted)
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
        f"ReproAgent: {run_error}" if run_error else
        "ReproAgent: budget exceeded with no submission" if crashed else
        "ReproAgent: failed to reproduce the bug"
    )
    return state
