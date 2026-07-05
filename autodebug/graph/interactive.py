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

import re
from typing import Annotated

# pydantic (used by langgraph to build the Studio context form) requires
# typing_extensions.TypedDict, not typing.TypedDict, on Python < 3.12.
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from autodebug.agents.manager import build_manager_agent
from autodebug.fsm import ManagerPhase
from autodebug.github import fetch_issue, repo_from_issue_url
from autodebug.graph.pipeline import clone_repo
from autodebug.state import DebugState, PipelineStage
from autodebug.telemetry import setup_tracing


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    debug: dict
    fsm_phase: str


class RunContext(TypedDict, total=False):
    """Run parameters for a served run. Declaring this as the graph's context schema
    makes LangGraph Studio render a config form for these fields. `repo_url` is the
    only required one; everything else is optional. (Agent Chat UI has no config
    panel, so there the repo URL is parsed from the chat message instead — see
    `_split_repo_and_report`.)"""
    repo_url: str
    ref: str
    known_good: str
    issue_url: str
    requirements: str
    setup_command: str
    python_version: str


def _latest_human_text(messages) -> str:
    for m in reversed(messages or []):
        role = m.get("type") or m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        if role in ("human", "user") or isinstance(m, HumanMessage):
            return str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))
    return ""


_URL_RE = re.compile(r"https?://[^\s>)\]}\"']+")


def _split_repo_and_report(text: str) -> tuple[str, str]:
    """Pull a repo URL out of a free-text chat message so a run can be driven from
    the message alone (no config panel — the only input Agent Chat UI offers).

    The first http(s) URL is taken as the repo; the rest is the bug report. A bare
    ``Repo:``/``Repository:`` label left behind is dropped. If there's no URL, the
    whole text is the bug report (repo must then come from the run config/context)."""
    text = text or ""
    m = _URL_RE.search(text)
    if not m:
        return "", text
    url = m.group(0).rstrip(".,)")
    report = _URL_RE.sub("", text, count=1)
    report = re.sub(r"(?im)^[ \t]*(repo(sitory)?|url)[ \t]*:[ \t]*$", "", report).strip()
    return url, (report or text)


def _ds(state) -> DebugState:
    return DebugState(**((state or {}).get("debug") or {"repo_url": "", "bug_report": ""}))


# --- nodes (sync; LangGraph runs them off the event loop) -------------------

def prepare(state: GraphState, config) -> dict:
    """Build the DebugState from run config + the chat's first human message."""
    setup_tracing()
    cfg = (config or {}).get("configurable", {}) or {}
    bug = _latest_human_text(state.get("messages")) or cfg.get("bug_report", "")
    issue_url = cfg.get("issue_url") or cfg.get("github_issue_url") or ""
    # repo_url comes from the run config/context if set; otherwise parse it out of the
    # chat message (the only input Agent Chat UI offers — paste the repo URL + report).
    repo_url = cfg.get("repo_url") or cfg.get("repo") or ""
    if not repo_url:
        parsed_url, bug = _split_repo_and_report(bug)
        # A pasted issue/PR URL doubles as both the issue AND (its parent) the repo.
        parent = repo_from_issue_url(parsed_url)
        if parent:
            repo_url = parent
            issue_url = issue_url or parsed_url
        else:
            repo_url = parsed_url
    # If an issue/PR URL was given (config or message), fold its title+body into the
    # bug report. Best-effort: fetch failure just leaves the report as-is.
    if issue_url:
        issue_text = fetch_issue(issue_url)
        if issue_text:
            bug = f"{bug}\n\n{issue_text}".strip() if bug else issue_text
    ds = DebugState(
        repo_url=repo_url,
        bug_report=bug,
        known_good_commit=cfg.get("known_good") or cfg.get("known_good_commit"),
        github_issue_url=issue_url or None,
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

    b = StateGraph(GraphState, context_schema=RunContext)
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
