"""Pipeline orchestrator — sequences clone → repro → bisect → root_cause → fix.

Each agent stage is a plain `run_<name>(state, *, registry) -> state` function
built with LangChain's `create_agent` (see autodebug/agents/). No LangGraph
node/edge wiring — short-circuit if any stage marks the state FAILED.

The repo lives in a Docker volume that is shared across all stages; the host
never touches the cloned files directly. This keeps symlinks intact and lets
the per-agent sandbox containers attach to the same repo state.
"""

from __future__ import annotations

from opentelemetry import trace

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
# Pipeline (a thin, unattended wrapper over THE graph — see registry.build_graph)
# ---------------------------------------------------------------------------


def run_pipeline(repo_url: str, bug_report: str, *, run_label: str | None = None,
                 **kwargs) -> DebugState:
    """Run AutoDebug unattended (no HITL) and return the final DebugState.

    This is a thin SYNCHRONOUS wrapper over THE graph — the exact same graph Studio
    and the CLI serve, just built with ``hitl=False`` — so eval/CLI/Studio share one
    orchestration. `run_label` names the root trace span (e.g. "ansible-1") for
    Phoenix; it never reaches the agents. Production inputs go in the run config; the
    agent graph never sees test metadata (that stays in the eval scorer).
    """
    setup_tracing()
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from autodebug.registry import AutoDebugRegistry

    graph = AutoDebugRegistry.from_file().build_graph(hitl=False, checkpointer=MemorySaver())
    config = {"configurable": {
        "thread_id": run_label or "autodebug",
        "repo_url": repo_url, "bug_report": bug_report,
        "ref": kwargs.get("ref"),
        "known_good": kwargs.get("known_good_commit"),
        "issue_url": kwargs.get("github_issue_url"),
        "requirements": kwargs.get("requirements"),
        "setup_command": kwargs.get("setup_command"),
        "python_version": kwargs.get("python_version"),
    }}
    state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
    with _tracer.start_as_current_span(run_label or "autodebug.pipeline") as span:
        span.set_attribute("repo_url", repo_url)
        span.set_attribute("bug_report", bug_report[:500])
        if run_label:
            span.set_attribute("instance_id", run_label)
        try:
            final = graph.invoke({"messages": [HumanMessage(content=bug_report)]}, config=config)
            debug = (final or {}).get("debug")
            if debug:
                state = DebugState(**debug)
            span.set_attribute("final_stage", str(state.stage))
            return state
        except Exception as e:  # noqa: BLE001 — surface as a FAILED state, don't crash
            span.record_exception(e)
            state.stage = PipelineStage.FAILED
            state.error = state.error or f"Pipeline: {type(e).__name__}: {str(e)[:300]}"
            return state
        finally:
            # Always release the volume so dangling state doesn't accumulate.
            if state.repo_volume:
                remove_repo_volume(state.repo_volume)
