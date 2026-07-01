"""The single AutoDebug graph — used by every entry point (eval, CLI, Studio,
Agent Chat UI).

Shape: ``prepare → clone → manager``, where ``manager`` is the Manager
``create_agent`` graph compiled WITHOUT its own checkpointer and added as a
**subgraph node**. Sharing the parent's checkpointer, its ``interrupt()``s (the two
HITL triggers) bubble to the served stream — Studio / Agent Chat UI render them and
resume continues the *same* conversation. Pipeline state + FSM phase live in the
shared ``debug`` / ``fsm_phase`` channels; ``messages`` is the chat surface.

There is ONE factory — ``AutoDebugRegistry.build_graph(hitl=…)`` — which delegates
to ``build()`` here. ``hitl`` is the only behavioral switch:
  - interactive (CLI/Studio): ``hitl=True``  → pauses for input when stuck.
  - unattended (eval/CI):      ``hitl=False`` → never blocks.

Nodes are plain SYNC functions: LangGraph runs them in its own executor under both
``.invoke()`` (eval) and ``.astream()`` (Studio), so the blocking Docker work never
stalls the event loop — which is exactly what lets one graph serve every route.

Exports: ``graph`` (hitl=True, no checkpointer) for ``langgraph dev``; ``build_graph``
for the CLI/tests.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from autodebug.agents.manager import build_manager_agent
from autodebug.fsm import ManagerPhase
from autodebug.graph.pipeline import clone_repo
from autodebug.state import DebugState, PipelineStage
from autodebug.telemetry import setup_tracing


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    debug: dict
    fsm_phase: str


def _latest_human_text(messages) -> str:
    for m in reversed(messages or []):
        role = m.get("type") or m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        if role in ("human", "user") or isinstance(m, HumanMessage):
            return str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))
    return ""


def _ds(state) -> DebugState:
    return DebugState(**((state or {}).get("debug") or {"repo_url": "", "bug_report": ""}))


# --- nodes (sync; LangGraph runs them off the event loop) -------------------

def prepare(state: GraphState, config) -> dict:
    """Build the DebugState from run config + the chat's first human message."""
    setup_tracing()
    cfg = (config or {}).get("configurable", {}) or {}
    bug = _latest_human_text(state.get("messages")) or cfg.get("bug_report", "")
    ds = DebugState(
        repo_url=cfg.get("repo_url") or cfg.get("repo") or "",
        bug_report=bug,
        known_good_commit=cfg.get("known_good") or cfg.get("known_good_commit"),
        github_issue_url=cfg.get("issue_url") or cfg.get("github_issue_url"),
        ref=cfg.get("ref"),
        requirements=cfg.get("requirements"),
        setup_command=cfg.get("setup_command"),
        python_version=cfg.get("python_version"),
    )
    return {
        "debug": ds.model_dump(),
        "fsm_phase": ManagerPhase.INIT.value,
        "messages": [AIMessage(content=f"🔍 Starting AutoDebug on {ds.repo_url or 'the repository'}.")],
    }


def clone(state: GraphState) -> dict:
    """Provision the sandbox volume + per-bug env (blocking Docker, off-loop)."""
    ds = _ds(state)
    try:
        ds = clone_repo(ds)
        msg = "📦 Cloned the repository and built its environment."
    except Exception as e:  # noqa: BLE001 — surface as FAILED, never crash the run
        ds.stage = PipelineStage.FAILED
        ds.error = f"sandbox setup failed: {type(e).__name__}: {str(e)[:300]}"
        msg = f"❌ {ds.error}"
    return {"debug": ds.model_dump(), "messages": [AIMessage(content=msg)]}


def _after_clone(state: GraphState) -> str:
    # Infra failure (e.g. Docker down) can't be fixed by the Manager — end.
    return END if str(_ds(state).stage) == PipelineStage.FAILED.value else "manager"


def record(state: GraphState) -> dict:
    """Persist the run as a memory episode (best-effort, never raises). Runs for
    EVERY route (CLI, Studio, eval), so the cross-bug memory corpus always grows;
    whether agents RECALL it is gated separately by AUTODEBUG_MEMORY_ENABLED."""
    from autodebug.memory import store_agent_run

    try:
        store_agent_run("manager", _ds(state))
    except Exception:  # noqa: BLE001 — memory must never affect the run
        pass
    return {}


# --- the single factory -----------------------------------------------------

def build(registry, *, hitl: bool = False, checkpointer=None, manager_node=None):
    """Compile THE graph. `manager_node` is injectable so tests stub the Manager
    (no LLM); production builds the real subgraph via build_manager_agent."""
    if manager_node is None:
        agent, _ = build_manager_agent(registry, checkpointer=None, hitl=hitl)
        manager_node = agent

    b = StateGraph(GraphState)
    b.add_node("prepare", prepare)
    b.add_node("clone", clone)
    b.add_node("manager", manager_node)
    b.add_node("record", record)
    b.add_edge(START, "prepare")
    b.add_edge("prepare", "clone")
    b.add_conditional_edges("clone", _after_clone, {"manager": "manager", END: END})
    b.add_edge("manager", "record")        # persist the episode after the Manager finishes
    b.add_edge("record", END)
    return b.compile(checkpointer=checkpointer)


# Served graph for `langgraph dev` / Studio / Agent Chat UI: HITL on, no checkpointer
# (the platform injects persistence). langgraph.json points here.
def _served_graph():
    from autodebug.registry import AutoDebugRegistry
    return AutoDebugRegistry.from_file().build_graph(hitl=True)


graph = _served_graph()


def build_graph(checkpointer=None, *, hitl: bool = True, manager_node=None):
    """Standalone-compiled instance for the CLI/tests (its own checkpointer). HITL on
    by default (interactive); pass hitl=False for unattended. Tests can stub the
    Manager via `manager_node`."""
    from autodebug.registry import AutoDebugRegistry

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    registry = AutoDebugRegistry.from_file()
    return build(registry, hitl=hitl, checkpointer=checkpointer, manager_node=manager_node)
