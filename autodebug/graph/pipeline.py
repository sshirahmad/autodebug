"""LangGraph pipeline — wires all agents into a checkpointed state machine."""

from __future__ import annotations

import tempfile
from typing import Literal

import git
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from autodebug.agents.bisect import BisectAgent
from autodebug.agents.fix import FixAgent
from autodebug.agents.repro import ReproAgent
from autodebug.agents.root_cause import RootCauseAgent
from autodebug.state import DebugState, PipelineStage


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def clone_repo(state: DebugState) -> DebugState:
    """Clone target repo into a temp directory with full history for bisect."""
    tmp = tempfile.mkdtemp(prefix="autodebug_")
    git.Repo.clone_from(state.repo_url, tmp)  # full clone — bisect needs history
    state.repo_local_path = tmp
    state.stage = PipelineStage.REPRO
    return state


def run_repro(state: DebugState) -> DebugState:
    return ReproAgent().run(state)


def run_bisect(state: DebugState) -> DebugState:
    return BisectAgent().run(state)


def run_root_cause(state: DebugState) -> DebugState:
    return RootCauseAgent().run(state)


def run_fix(state: DebugState) -> DebugState:
    return FixAgent().run(state)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_repro(state: DebugState) -> Literal["bisect", "failed"]:
    return "bisect" if state.stage == PipelineStage.BISECT else "failed"


def route_after_bisect(state: DebugState) -> Literal["root_cause", "failed"]:
    return "root_cause" if state.stage == PipelineStage.ROOT_CAUSE else "failed"


def route_after_root_cause(state: DebugState) -> Literal["fix", "failed"]:
    return "fix" if state.stage == PipelineStage.FIX else "failed"


def route_after_fix(state: DebugState) -> Literal["done", "failed"]:
    return "done" if state.stage == PipelineStage.DONE else "failed"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    graph = StateGraph(DebugState)

    graph.add_node("clone", clone_repo)
    graph.add_node("repro", run_repro)
    graph.add_node("bisect", run_bisect)
    graph.add_node("root_cause", run_root_cause)
    graph.add_node("fix", run_fix)

    graph.set_entry_point("clone")
    graph.add_edge("clone", "repro")

    graph.add_conditional_edges("repro", route_after_repro, {
        "bisect": "bisect",
        "failed": END,
    })
    graph.add_conditional_edges("bisect", route_after_bisect, {
        "root_cause": "root_cause",
        "failed": END,
    })
    graph.add_conditional_edges("root_cause", route_after_root_cause, {
        "fix": "fix",
        "failed": END,
    })
    graph.add_conditional_edges("fix", route_after_fix, {
        "done": END,
        "failed": END,
    })

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_pipeline(repo_url: str, bug_report: str, **kwargs) -> DebugState:
    """Entry point: run the full pipeline and return final state."""
    app = build_graph()
    initial_state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
    config = {"configurable": {"thread_id": "main"}}
    return app.invoke(initial_state, config=config)
