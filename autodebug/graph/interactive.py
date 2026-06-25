"""Interactive top-level graph — a thin composition over the existing pipeline.

This turns the imperative `run_pipeline` into a compiled LangGraph `StateGraph`
that adds exactly three things the blocking pipeline lacks:

  1. a ``messages`` channel, so Agent Chat UI (or any LangGraph client) can render
     a live conversation as the run progresses;
  2. streamed per-stage progress; and
  3. a human-in-the-loop ``human_review`` node that ``interrupt()``s on failure and
     resumes with the developer's feedback.

It does NOT re-implement orchestration. The stage nodes DELEGATE to the existing
``run_*`` runners (autodebug/agents/), which already build and invoke
``create_agent``'s compiled sub-graphs. Each sub-agent keeps its OWN checkpointer
threads (see autodebug/resume.py); the parent saver is never pushed down into
them. That separation is deliberate: only THIS top graph's ``human_review`` node
calls ``interrupt()``, so the interrupt bubbles up to the served stream while the
sub-agents run to completion under their own savers.

Two compiled forms are exported:
  - ``graph`` — compiled WITHOUT a checkpointer, for ``langgraph dev`` / the
    platform (which injects persistence). This is what ``langgraph.json`` points at
    and what Agent Chat UI connects to.
  - ``build_graph(checkpointer)`` — a standalone instance WITH a checkpointer, for
    the CLI ``--stream`` path (and tests), where we manage threads in-process
    rather than under the dev server.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from autodebug.agents.base import set_stream_tokens
from autodebug.graph.pipeline import _failed, _resolve_stages, clone_repo
from autodebug.memory import store_agent_run
from autodebug.state import DebugState, PipelineStage
from autodebug.telemetry import setup_tracing


class GraphState(TypedDict, total=False):
    """Top-graph channels. ``messages`` is the chat surface Agent Chat UI renders;
    ``debug`` is the real pipeline state threaded through the stage runners."""

    messages: Annotated[list[AnyMessage], add_messages]
    debug: DebugState
    user_feedback: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(text: str) -> None:
    """Best-effort custom progress event (ignored if there's no active writer),
    so richer UIs (Studio / Agent Chat UI custom events, CLI) can show fine-grained
    progress."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"progress": text})
    except Exception:
        pass


def _message_text(m) -> str:
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def _latest_human_text(messages) -> str:
    """The most recent human turn — the bug report when driven from a chat box."""
    for m in reversed(messages or []):
        role = m.get("type") or m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        if role in ("human", "user") or isinstance(m, HumanMessage):
            return _message_text(m)
    return ""


def _stage_recap(d: DebugState) -> str:
    """One line per stage that produced an artifact — the 'what was tried' recap."""
    parts = []
    if d.repro:
        parts.append(f"• repro: {'✅ reproduced' if d.repro.confirmed else '⚠️ not confirmed'}")
    if d.bisect and d.bisect.culprit_commit:
        parts.append(f"• bisect: culprit {d.bisect.culprit_commit[:8]} — {d.bisect.commit_message[:60]}")
    if d.root_cause:
        parts.append(f"• root cause: {d.root_cause.summary[:120]}")
    if d.hypothesis_attempts:
        tried = "; ".join(
            f"{a.get('hypothesis', '?')[:50]}→{a.get('outcome', '?')}"
            for a in d.hypothesis_attempts
        )
        parts.append(f"• fix attempts: {tried}")
    return "\n".join(parts) or "• no stage produced an artifact yet"


def _summary_message(d: DebugState) -> str:
    if d.stage == PipelineStage.DONE and d.fix:
        head = "✅ Fixed and verified against the reproduction."
        if d.fix.pr_url:
            head += f"\nPR: {d.fix.pr_url}"
        return f"{head}\n\n{_stage_recap(d)}"
    return f"⚠️ Did not converge on a verified fix.\n\n{_stage_recap(d)}\n\nError: {d.error or 'unknown'}"


def _failure_summary(d: DebugState) -> str:
    """The interrupt payload shown to the developer: what happened, what was tried,
    and what they can do. A readable string so any LangGraph client renders it."""
    return (
        "AutoDebug got stuck and needs your input.\n\n"
        f"What happened: {d.error or 'no verified fix produced'}\n\n"
        f"Progress so far:\n{_stage_recap(d)}\n\n"
        "Reply with guidance to steer the next attempt — e.g. a file/function to "
        "focus on, a hypothesis to try, a different repro approach, or `skip` to give "
        "up. Your message is fed straight into the next attempt."
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def prepare(state: GraphState, config) -> dict:
    """Build the DebugState from the run config + the chat's first human message."""
    # setup_tracing() imports Phoenix, which does blocking filesystem I/O on first
    # call — run it off the event loop so the ASGI server (langgraph dev) isn't
    # flagged/stalled by a blocking call.
    await asyncio.to_thread(setup_tracing)
    cfg = (config or {}).get("configurable", {}) or {}
    bug = _latest_human_text(state.get("messages")) or cfg.get("bug_report", "")
    debug = DebugState(
        repo_url=cfg.get("repo_url") or cfg.get("repo") or "",
        bug_report=bug,
        known_good_commit=cfg.get("known_good") or cfg.get("known_good_commit"),
        github_issue_url=cfg.get("issue_url") or cfg.get("github_issue_url"),
        ref=cfg.get("ref"),
        requirements=cfg.get("requirements"),
        setup_command=cfg.get("setup_command"),
        python_version=cfg.get("python_version"),
    )
    target = debug.repo_url or "the repository"
    return {
        "debug": debug,
        "messages": [AIMessage(content=f"🔍 Starting AutoDebug on {target}.")],
    }


async def clone(state: GraphState) -> dict:
    """Provision the sandbox volume + per-bug env (blocking Docker → a thread)."""
    debug = state["debug"]
    try:
        debug = await asyncio.to_thread(clone_repo, debug)
        msg = "📦 Cloned the repository and built its environment."
    except Exception as e:  # noqa: BLE001 — surface as a FAILED state, never crash the run
        debug.stage = PipelineStage.FAILED
        debug.error = f"sandbox setup failed: {type(e).__name__}: {str(e)[:300]}"
        msg = f"❌ {debug.error}"
    _emit(msg)
    return {"debug": debug, "messages": [AIMessage(content=msg)]}


async def solve(state: GraphState, config) -> dict:
    """Run the resolved stages (Manager FSM by default, else linear) in a worker
    thread. Identical orchestration to run_pipeline — reused, not re-implemented."""
    # Per-run token streaming: set the contextvar BEFORE to_thread so the copied
    # worker-thread context (where the sub-agents build their models) sees it.
    cfg = (config or {}).get("configurable", {}) or {}
    set_stream_tokens(bool(cfg.get("stream_tokens")))

    debug = state["debug"]
    feedback = state.get("user_feedback")
    if feedback:
        debug.user_feedback = feedback
        # This is a retry after HITL. The previous attempt left the state FAILED,
        # which the stage loop below treats as "stop" — so clear that marker and
        # re-arm the pipeline, otherwise the run bounces straight back to
        # human_review without ever re-running the stages with the new guidance.
        if _failed(debug):
            debug.stage = PipelineStage.REPRO
            debug.error = None

    def _run() -> DebugState:
        # Build the registry INSIDE the thread: AutoDebugRegistry.from_file() reads
        # config off disk (blocking), so it must not run on the event loop.
        from autodebug.registry import AutoDebugRegistry

        registry = AutoDebugRegistry.from_file()
        d = debug
        for name, runner in _resolve_stages(registry):
            if _failed(d):
                break
            _emit(f"▶️ {name}…")
            d = runner(d, registry=registry)
            store_agent_run(name, d)  # best-effort memory write (recall gated separately)
        return d

    debug = await asyncio.to_thread(_run)
    msg = _summary_message(debug)
    _emit(msg)
    # Clear consumed feedback so a later replay doesn't re-apply stale guidance.
    return {"debug": debug, "user_feedback": None, "messages": [AIMessage(content=msg)]}


def human_review(state: GraphState) -> dict:
    """HITL: pause on failure, show a summary, resume with the developer's feedback.

    This is the ONLY node that interrupts, so the pause surfaces on the served
    stream. The client resumes with ``Command(resume=<feedback>)``; the feedback is
    written onto ``user_feedback`` and a static edge loops back to ``solve``. (We
    deliberately don't ALSO return a ``Command(goto=...)`` here — mixing an
    ``interrupt()`` with a routing Command in one node makes the resume re-interrupt
    instead of proceeding.)
    """
    feedback = interrupt(_failure_summary(state["debug"]))
    if isinstance(feedback, dict):
        feedback = feedback.get("feedback") or feedback.get("content") or ""
    feedback = str(feedback or "").strip()
    return {
        "user_feedback": feedback,
        "messages": [HumanMessage(content=f"[developer feedback] {feedback}")],
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _after_clone(state: GraphState) -> str:
    # An infra failure (e.g. Docker down) can't be fixed by human guidance, so end.
    return END if _failed(state["debug"]) else "solve"


def _after_solve(state: GraphState) -> str:
    return "human_review" if _failed(state["debug"]) else END


def _builder() -> StateGraph:
    b = StateGraph(GraphState)
    b.add_node("prepare", prepare)
    b.add_node("clone", clone)
    b.add_node("solve", solve)
    b.add_node("human_review", human_review)

    b.add_edge(START, "prepare")
    b.add_edge("prepare", "clone")
    b.add_conditional_edges("clone", _after_clone, {"solve": "solve", END: END})
    b.add_conditional_edges("solve", _after_solve, {"human_review": "human_review", END: END})
    b.add_edge("human_review", "solve")  # after feedback, retry the solve loop
    return b


# For `langgraph dev` / the platform (it injects persistence) and Agent Chat UI.
graph = _builder().compile()


def build_graph(checkpointer=None):
    """A standalone-compiled instance WITH a checkpointer, for the CLI ``--stream``
    path (and tests) where we manage threads in-process (the dev server isn't there
    to provide persistence). Defaults to an in-process MemorySaver."""
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return _builder().compile(checkpointer=checkpointer)
