"""LangGraph pipeline — wires all agents into a checkpointed state machine."""

from __future__ import annotations

import tempfile
from typing import Literal

import git
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from autodebug.telemetry import setup_tracing
from autodebug.state import DebugState, PipelineStage
from autodebug.memory import store_agent_run


def _coerce(state) -> DebugState:
    return DebugState(**state) if isinstance(state, dict) else state


def clone_repo(state) -> dict:
    state = _coerce(state)
    tmp = tempfile.mkdtemp(prefix="autodebug_")
    repo = git.Repo.clone_from(state.repo_url, tmp)
    if state.pre_fix_commit:
        repo.git.checkout(state.pre_fix_commit)
    state.repo_local_path = tmp
    state.stage = PipelineStage.REPRO
    return state.model_dump()


def route_after_repro(state) -> Literal["bisect", "failed"]:
    return "bisect" if _coerce(state).stage == PipelineStage.BISECT else "failed"


def route_after_bisect(state) -> Literal["root_cause", "failed"]:
    return "root_cause" if _coerce(state).stage == PipelineStage.ROOT_CAUSE else "failed"


def route_after_root_cause(state) -> Literal["fix", "failed"]:
    return "fix" if _coerce(state).stage == PipelineStage.FIX else "failed"


def route_after_fix(state) -> Literal["done", "failed"]:
    return "done" if _coerce(state).stage == PipelineStage.DONE else "failed"


def build_graph(registry, checkpointer=None):
    graph = StateGraph(DebugState)

    repro      = registry.create_repro_agent()
    bisect     = registry.create_bisect_agent()
    root_cause = registry.create_root_cause_agent()
    fix        = registry.create_fix_agent()

    graph.add_node("clone",      clone_repo)
    def _run(stage: str, agent, s):
        state = agent.run(_coerce(s))
        store_agent_run(stage, state)
        return state.model_dump()

    graph.add_node("repro",      lambda s: _run("repro", repro, s))
    graph.add_node("bisect",     lambda s: _run("bisect", bisect, s))
    graph.add_node("root_cause", lambda s: _run("root_cause", root_cause, s))
    graph.add_node("fix",        lambda s: _run("fix", fix, s))

    graph.set_entry_point("clone")
    graph.add_edge("clone", "repro")

    graph.add_conditional_edges("repro", route_after_repro, {"bisect": "bisect", "failed": END})
    graph.add_conditional_edges("bisect", route_after_bisect, {"root_cause": "root_cause", "failed": END})
    graph.add_conditional_edges("root_cause", route_after_root_cause, {"fix": "fix", "failed": END})
    graph.add_conditional_edges("fix", route_after_fix, {"done": END, "failed": END})

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_pipeline(repo_url: str, bug_report: str, **kwargs) -> DebugState:
    from autodebug.registry import AutoDebugRegistry
    setup_tracing()
    registry = AutoDebugRegistry.from_file()
    app = build_graph(registry)
    initial_state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "main"}})
    return _coerce(result)
