"""Root Cause Agent — explains WHY the bug happens, backed by observed evidence.

It submits via `submit_root_cause` (which requires an `evidence` arg). The driver
nudges it to actually observe the failure: if it submits without ever calling an
inspection tool (run_repro_with_traceback / inspect_at) and retries remain, the
submission is rejected and it's told to observe first. On the final attempt a
submission is accepted regardless, so the pipeline degrades gracefully rather
than failing outright.

(We use a submit tool rather than create_agent's response_format because the
OpenRouter routing for our model rejects the tool_choice that structured output
relies on.)
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from autodebug.agents.base import (
    Budget, BudgetExceeded, attempt_trajectory, budget_middleware,
    budget_nudge_middleware, maybe_optimize_prompt, model_for_attempt,
    model_retry_middleware, planning_middleware, require_tool_calls_middleware,
    retry_feedback, submission_middleware, summarization_middleware,
    tool_call_limit_middleware,
)
from autodebug import resume
from autodebug.agent_state import RootCauseAgentState
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, RootCauseResult

_INSPECTION_TOOLS = {"run_repro_with_traceback", "inspect_at"}


def _used_inspection_tool(messages: list) -> bool:
    """Did the agent actually observe the failure at runtime (not just read code)?"""
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name") in _INSPECTION_TOOLS:
                    return True
    return False


def run_root_cause(state: DebugState, *, registry) -> DebugState:
    """Drive the root_cause agent until it submits an evidence-backed hypothesis."""
    assert state.repo_volume
    assert state.repro  # bisect (culprit) is best-effort; root_cause works without it

    cfg = registry.get_config("root_cause")

    system_prompt = registry.system_prompt("root_cause")

    # Refine rather than regenerate: a prior hypothesis means the fix based on it
    # failed, so the manager looped back. Surface it so the agent corrects what was
    # wrong instead of re-deriving (and possibly re-deriving the same mistake).
    refine_note = ""
    if state.root_cause is not None:
        refine_note = (
            "\nNOTE: a previous root-cause hypothesis was produced but the fix based "
            "on it FAILED verification, so it was wrong or incomplete. Previous "
            f"hypothesis:\n{state.root_cause.hypothesis}\n{state.root_cause.summary[:800]}\n"
            "Identify what that analysis got wrong and produce a corrected hypothesis "
            "(in particular, distinguish the real defect from any environment/import "
            "artifact in the sandbox).\n"
        )

    crashed = False
    run_error: str | None = None
    key = resume.bug_key(state)
    saver = resume.get_saver()

    # Resume: a prior run already produced a root cause (pure reasoning — no
    # sandbox needed to rebuild it). Skip straight to the fix.
    cached = resume.cached_root_cause(key, cfg.max_retries)
    if cached is not None:
        state.root_cause = cached
        state.stage = PipelineStage.FIX
        return state
    resume.clear("root_cause", key, cfg.max_retries)

    with Sandbox(volume=state.repo_volume) as sandbox:
        initial_run = sandbox.run_script(state.repro.repro_script)
        if state.bisect:
            culprit_block = (
                f"Culprit commit: {state.bisect.commit_message}\n"
                f"SHA: {state.bisect.culprit_commit}\n\n"
                f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
            )
        else:
            culprit_block = (
                "Culprit commit: NOT identified (bisect was inconclusive). Work from "
                "the bug report, the reproduction, and the live failure instead — "
                "`read_file_at_parent` is unavailable, so rely on "
                "run_repro_with_traceback, inspect_at, and read_file.\n\n"
            )
        initial_text = (
            f"Bug report (the symptom your hypothesis MUST explain):\n"
            f"{state.bug_report}\n\n"
            f"{culprit_block}"
            f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n"
            f"{refine_note}\n"
            "Identify the root cause of the bug described above. You MUST observe "
            "the failure at runtime — call run_repro_with_traceback or inspect_at — "
            "and base your `evidence` on what you actually saw, before calling "
            "submit_root_cause."
        )

        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            llm = model_for_attempt(attempt, cfg.model, cfg.provider)
            tools = registry.build_tools(
                "root_cause",
                sandbox=sandbox,
                parent_sha=f"{state.bisect.culprit_commit}^" if state.bisect else "",
                repro_script=state.repro.repro_script,
            )
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                state_schema=RootCauseAgentState,
                checkpointer=saver,
                middleware=(
                    budget_middleware(budget)
                    + budget_nudge_middleware(budget, "submit_root_cause")
                    + model_retry_middleware()
                    + planning_middleware()
                    + summarization_middleware(cfg.model, cfg.provider)
                    + submission_middleware("root_cause")
                    + require_tool_calls_middleware()
                    + tool_call_limit_middleware(cfg.tool_call_limits)
                ),
            )
            invoke_config = {
                "recursion_limit": sys.maxsize,
                "configurable": {"thread_id": resume.thread_id("root_cause", key, attempt)},
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

            submitted = resume.read_live(agent, invoke_config, "root_cause")
            if submitted:
                messages = attempt_trajectory(agent, invoke_config)
                last = attempt < cfg.max_retries
                if _used_inspection_tool(messages) or not last:
                    # Accept if it observed the failure — or on the final attempt,
                    # accept anyway so the pipeline isn't blocked by a missing probe.
                    state.root_cause = RootCauseResult(**submitted)
                    state.stage = PipelineStage.FIX
                    return state
                # Submitted without observing AND retries remain: reject and drop
                # this attempt's thread so the unobserved hypothesis can't be resumed.
                resume.clear_one("root_cause", key, attempt)
                run_error = "submitted a hypothesis without observing the failure"
                system_prompt += (
                    "\n\nYOU SUBMITTED WITHOUT OBSERVING THE FAILURE. Before calling "
                    "submit_root_cause you MUST call run_repro_with_traceback or "
                    "inspect_at and base `evidence` on its actual output."
                )

            if attempt >= cfg.max_retries:
                break

            # Retrying (crash or rejected): optimize the prompt from this attempt.
            system_prompt = maybe_optimize_prompt(
                system_prompt,
                attempt_trajectory(agent, invoke_config),
                retry_feedback("determining the root cause with observed evidence"),
                model_id=cfg.model, provider=cfg.provider,
            )

    state.stage = PipelineStage.FAILED
    state.error = (
        f"RootCauseAgent: {run_error}" if run_error else
        "RootCauseAgent: budget exceeded with no submission" if crashed else
        "RootCauseAgent: could not determine root cause"
    )
    return state
