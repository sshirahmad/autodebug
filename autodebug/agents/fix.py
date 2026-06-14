"""Fix Agent — generates and validates patches."""

from __future__ import annotations

import os
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
from autodebug.agent_state import FixAgentState
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, FixResult, PipelineStage


def run_fix(state: DebugState, *, registry) -> DebugState:
    """Drive the fix agent until it submits a passing patch."""
    assert state.repo_volume
    assert state.repro
    assert state.bisect
    assert state.root_cause

    cfg = registry.get_config("fix")

    rc = state.root_cause
    plan_block = (
        f"## FIX PLAN — implement this exactly (root cause already determined it)\n"
        f"{rc.fix_plan}\n\n"
        if rc.fix_plan else
        # Fallback if root cause produced no explicit plan.
        f"## Root cause\n{rc.hypothesis}\n"
        + "Relevant lines:\n" + "\n".join(f"  - {l}" for l in rc.relevant_lines) + "\n\n"
    )
    evidence_block = (
        f"## Observed failure (from root cause)\n{rc.evidence}\n\n" if rc.evidence else ""
    )
    initial_text = (
        plan_block
        + evidence_block
        + "## Success criterion — the reproduction MUST exit 0 after your fix\n"
          f"```python\n{state.repro.repro_script}\n```\n\n"
        + "Implement the FIX PLAN above with `apply_patch`, then run `run_repro` to "
          "verify. Do NOT re-investigate the bug — the root cause is settled. Read a "
          "file only to get the exact text for a patch. If `run_repro` still fails, "
          "read its traceback/state and adjust your patch to match the plan."
    )

    model_id = cfg.model or os.getenv("AUTODEBUG_FIX_MODEL")
    provider = cfg.provider or os.getenv("AUTODEBUG_FIX_MODEL_PROVIDER")
    system_prompt = registry.system_prompt("fix")
    llm = build_model(model_id=model_id, provider=provider)

    crashed = False
    run_error: str | None = None
    key = resume.bug_key(state)
    saver = resume.get_saver()
    # The fix is never resumed from cache — it's the stage we re-attempt — but its
    # state is still persisted (durability). Drop stale threads before a fresh run.
    resume.clear("fix", key, cfg.max_retries)
    with Sandbox(volume=state.repo_volume) as sandbox:
        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            tools = registry.build_tools(
                "fix",
                sandbox=sandbox,
                repro_script=state.repro.repro_script,
            )
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                state_schema=FixAgentState,
                checkpointer=saver,
                middleware=(
                    budget_middleware(budget)
                    + model_retry_middleware()
                    + planning_middleware()
                    + summarization_middleware(model_id, provider)
                    + submission_middleware("fix")
                    + require_tool_calls_middleware()
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": resume.thread_id("fix", key, attempt)},
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

            submitted = resume.read_live(agent, invoke_config, "fix")
            if submitted:
                patches = resume.read_live(agent, invoke_config, "patches") or []
                state.fix = FixResult(**{**submitted, "attempts": len(patches)})
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
        f"FixAgent: {run_error}" if run_error else
        "FixAgent: budget exceeded with no submission" if crashed else
        "FixAgent: could not produce a valid fix"
    )
    return state
