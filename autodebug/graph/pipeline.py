"""Pipeline orchestrator — sequences clone → repro → bisect → root_cause → fix.

Each agent stage is a plain `run_<name>(state, *, registry) -> state` function
built with LangChain's `create_agent` (see autodebug/agents/). No LangGraph
node/edge wiring — short-circuit if any stage marks the state FAILED.

The repo lives in a Docker volume that is shared across all stages; the host
never touches the cloned files directly. This keeps symlinks intact and lets
the per-agent sandbox containers attach to the same repo state.
"""

from __future__ import annotations

import os

from opentelemetry import trace

from autodebug.agents import run_bisect, run_fix, run_repro, run_root_cause
from autodebug.memory import store_agent_run
from autodebug.sandbox import (
    clone_into_volume,
    create_repo_volume,
    remove_repo_volume,
)
from autodebug.state import DebugState, PipelineStage
from autodebug.telemetry import setup_tracing

_tracer = trace.get_tracer("autodebug.pipeline")


def clone_repo(state: DebugState) -> DebugState:
    """Provision a Docker volume and clone the repo into it via a one-shot container."""
    volume = create_repo_volume()
    try:
        # Production clone: just the repo at the given ref (default HEAD). No test
        # files are injected — the benchmark's FAIL_TO_PASS test is applied only in
        # the eval harness's separate scoring sandbox, never in the agents' repo.
        clone_into_volume(
            volume, state.repo_url, state.ref,
            requirements=state.requirements, setup_command=state.setup_command,
            python_version=state.python_version,
        )
    except Exception:
        remove_repo_volume(volume)
        raise
    state.repo_volume = volume
    state.stage = PipelineStage.REPRO
    return state


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_STAGES = (
    ("repro", run_repro),
    ("bisect", run_bisect),
    ("root_cause", run_root_cause),
    ("fix", run_fix),
)


def _resolve_stages(registry):
    """Pick the orchestration mode.

    If a `manager` agent is configured (and not disabled via AUTODEBUG_MANAGER=0),
    run the single FSM-driven Manager agent, which delegates to the sub-agents
    itself. Otherwise fall back to the classic linear repro->...->fix sequence.
    """
    manager_on = (
        "manager" in registry.config.agents
        and os.getenv("AUTODEBUG_MANAGER", "1") != "0"
    )
    if manager_on:
        from autodebug.agents import run_manager
        return (("manager", run_manager),)
    return _STAGES


def _failed(state: DebugState) -> bool:
    # state.stage may be enum or its string value depending on coercion path.
    return str(state.stage) in (PipelineStage.FAILED.value, str(PipelineStage.FAILED))


def run_pipeline(repo_url: str, bug_report: str, *, run_label: str | None = None,
                 **kwargs) -> DebugState:
    """Run the full clone → repro → bisect → root_cause → fix sequence.

    `run_label` names the root trace span (e.g. the bug id, "ansible-1") so runs
    are identifiable in Phoenix; it's observability-only and never reaches the
    agents. Defaults to "autodebug.pipeline".
    """
    setup_tracing()
    from autodebug.registry import AutoDebugRegistry
    registry = AutoDebugRegistry.from_file()

    with _tracer.start_as_current_span(run_label or "autodebug.pipeline") as span:
        span.set_attribute("repo_url", repo_url)
        span.set_attribute("bug_report", bug_report[:500])
        if run_label:
            span.set_attribute("instance_id", run_label)

        state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
        try:
            state = clone_repo(state)

            for name, runner in _resolve_stages(registry):
                if _failed(state):
                    break
                with _tracer.start_as_current_span(f"autodebug.{name}"):
                    state = runner(state, registry=registry)
                # Always accumulate memories (best-effort, never raises). Whether
                # agents RECALL them is gated separately by AUTODEBUG_MEMORY_ENABLED
                # (see make_search_memory_tool), so the corpus grows even when recall
                # is off for a clean/reproducible baseline.
                store_agent_run(name, state)

            span.set_attribute("final_stage", str(state.stage))
            return state
        except Exception as e:  # noqa: BLE001 — surface as a FAILED state, don't crash
            # Infra/transient failures (e.g. Docker not running, an unhandled
            # provider error) should produce a clean FAILED result, not abort the
            # caller. Record the span error for observability, then return state.
            span.record_exception(e)
            state.stage = PipelineStage.FAILED
            state.error = state.error or f"Pipeline: {type(e).__name__}: {str(e)[:300]}"
            return state
        finally:
            # Always release the volume so dangling state doesn't accumulate.
            if state.repo_volume:
                remove_repo_volume(state.repo_volume)
