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
        clone_into_volume(
            volume,
            state.repo_url,
            state.pre_fix_commit,
            test_patch=state.test_patch,
            fixed_commit=state.fixed_commit_id,
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


def run_pipeline(repo_url: str, bug_report: str, **kwargs) -> DebugState:
    """Run the full clone → repro → bisect → root_cause → fix sequence."""
    setup_tracing()
    from autodebug.registry import AutoDebugRegistry
    registry = AutoDebugRegistry.from_file()

    with _tracer.start_as_current_span("autodebug.pipeline") as span:
        span.set_attribute("repo_url", repo_url)
        span.set_attribute("bug_report", bug_report[:500])

        state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
        try:
            state = clone_repo(state)

            for name, runner in _resolve_stages(registry):
                if _failed(state):
                    break
                with _tracer.start_as_current_span(f"autodebug.{name}"):
                    state = runner(state, registry=registry)
                if os.getenv("AUTODEBUG_MEMORY_ENABLED", "0") == "1":
                    store_agent_run(name, state)

            span.set_attribute("final_stage", str(state.stage))
            return state
        finally:
            # Always release the volume so dangling state doesn't accumulate.
            if state.repo_volume:
                remove_repo_volume(state.repo_volume)
