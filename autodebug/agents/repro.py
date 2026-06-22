"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from autodebug.agents.base import (
    Budget, BudgetExceeded, attempt_trajectory, budget_middleware,
    maybe_audit, maybe_optimize_prompt, model_for_attempt, model_retry_middleware,
    planning_middleware, require_tool_calls_middleware, retry_feedback,
    submission_middleware, summarization_middleware, tool_call_limit_middleware,
)
from autodebug import resume
from autodebug.agent_state import ReproAgentState
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, ReproResult
from autodebug.tools import git_utils


def run_repro(state: DebugState, *, registry) -> DebugState:
    """Drive the repro agent until it produces a confirmed reproducing script."""
    assert state.repo_volume
    cfg = registry.get_config("repro")
    key = resume.bug_key(state)

    # Don't restart blind: carry forward context from the previous invocation.
    refine_note = ""
    if state.repro is not None:
        # A reproduction was CONFIRMED but the downstream fix couldn't be verified
        # against it (manager looped back) — hand back the prior script to refine.
        refine_note = (
            "\nNOTE: a previous reproduction was confirmed but the downstream fix "
            "could not be verified against it — it may be imprecise or target the "
            "wrong failure. Previous reproduction:\n"
            f"```python\n{state.repro.repro_script}\n```\n"
            f"Previous observed failure:\n{state.repro.error_output[:1000]}\n"
            "Refine THIS reproduction so it precisely targets the bug described "
            "above; do not start from an unrelated angle.\n"
        )
    else:
        # A previous attempt FAILED to confirm a repro. If it at least tried a
        # candidate script, surface it so this attempt diagnoses that dead-end
        # instead of repeating it.
        prior = resume.last_attempt_artifact("repro", key, cfg.max_retries, "submit_repro")
        if prior and (prior.get("script") or "").strip():
            refine_note = (
                "\nNOTE: a PREVIOUS attempt did not produce a confirmed reproduction. "
                "It tried this script, which did NOT reliably trigger the reported bug:\n"
                f"```python\n{prior['script']}\n```\n"
                "Work out WHY it failed and take a different approach — do not just "
                "resubmit the same script.\n"
            )
    initial_text = (
        f"Bug report:\n\n{state.bug_report}\n\n"
        f"{refine_note}"
        "Reproduce this bug: explore the repository, then write a minimal Python "
        "script that triggers the failure described above and confirm it fails.\n"
    )

    system_prompt = registry.system_prompt("repro")

    crashed = False
    run_error: str | None = None
    saver = resume.get_saver()
    with Sandbox(volume=state.repo_volume) as sandbox:
        # Run against the clean buggy tree: a manager loop-back may follow a failed
        # fix that left patches on the shared volume.
        git_utils.reset_worktree(sandbox)
        # Resume: a prior run already produced (and we re-validate) a repro.
        cached = resume.cached_repro(key, cfg.max_retries, sandbox)
        if cached is not None:
            state.repro = cached
            state.stage = PipelineStage.BISECT
            return state
        resume.clear("repro", key, cfg.max_retries)  # drop stale threads before a fresh run

        for attempt in range(cfg.max_retries + 1):
            budget = Budget.from_config(cfg)
            llm = model_for_attempt(attempt, cfg.model, cfg.provider)
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
                last = attempt >= cfg.max_retries
                # Adversarial audit: a confirmed repro can still be a weak oracle.
                # On the final attempt accept regardless, so the judge can't block
                # the pipeline outright.
                ok, critique = (True, "") if last else maybe_audit(
                    "repro",
                    f"Bug report:\n{state.bug_report}\n\n"
                    f"Reproduction script:\n{submitted.get('repro_script', '')}\n\n"
                    "Observed failure on the buggy code:\n"
                    f"{submitted.get('error_output', '')[:1500]}",
                    model_id=cfg.model, provider=cfg.provider,
                )
                if ok:
                    state.repro = ReproResult(**submitted)
                    state.stage = PipelineStage.BISECT
                    return state
                # Judge flagged a weak repro and retries remain: drop this thread
                # and retry with the critique so the next attempt strengthens it.
                resume.clear_one("repro", key, attempt)
                run_error = f"reproduction rejected by audit: {critique}"
                system_prompt += (
                    "\n\nA REVIEWER REJECTED YOUR REPRODUCTION as too weak to be a "
                    f"reliable oracle:\n{critique}\nStrengthen it to address this "
                    "(assert the full expected behavior, add discriminating cases) "
                    "before submitting again."
                )
                continue

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
