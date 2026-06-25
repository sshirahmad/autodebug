"""Tests for the interactive top-level graph (autodebug/graph/interactive.py).

These exercise the composition graph WITHOUT Docker or real agents: the sandbox
clone and the stage runners are monkeypatched, so we verify the wiring — the
messages channel, the happy path, failure routing to human_review, and that a
HITL resume injects the developer's feedback and loops back to solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.graph import interactive as gapp  # noqa: E402
from autodebug.state import DebugState, FixResult, PipelineStage, ReproResult  # noqa: E402


# --- helpers -------------------------------------------------------------

class TestHelpers:
    def test_latest_human_text_prefers_last_human_turn(self):
        from langchain_core.messages import AIMessage

        msgs = [HumanMessage(content="first"), AIMessage(content="mid"),
                HumanMessage(content="the bug report")]
        assert gapp._latest_human_text(msgs) == "the bug report"

    def test_stage_recap_lists_confirmed_repro(self):
        d = DebugState(repo_url="u", bug_report="b",
                       repro=ReproResult(repro_script="s", error_output="e", confirmed=True))
        assert "repro: ✅ reproduced" in gapp._stage_recap(d)

    def test_failure_summary_includes_error_and_asks_for_guidance(self):
        d = DebugState(repo_url="u", bug_report="b", error="no fix")
        out = gapp._failure_summary(d)
        assert "no fix" in out and "guidance" in out.lower()


# --- graph wiring --------------------------------------------------------

def _patch_clone(monkeypatch):
    """clone succeeds: sets a volume and advances to REPRO (like clone_repo)."""
    def fake_clone(state: DebugState) -> DebugState:
        state.repo_volume = "vol-test"
        state.stage = PipelineStage.REPRO
        return state

    monkeypatch.setattr(gapp, "clone_repo", fake_clone)
    monkeypatch.setattr(gapp, "store_agent_run", lambda *a, **k: None)
    # solve builds a registry; make it cheap and inert.
    import autodebug.registry as reg
    monkeypatch.setattr(reg.AutoDebugRegistry, "from_file", classmethod(lambda cls: object()))


def _patch_stage(monkeypatch, runner):
    monkeypatch.setattr(gapp, "_resolve_stages", lambda registry: (("manager", runner),))


def _cfg():
    return {"configurable": {"thread_id": "t1"}}


def test_graph_exposes_messages_channel():
    # Agent Chat UI requires a `messages` key in graph state.
    assert "messages" in gapp.graph.channels


async def test_happy_path_runs_to_done(monkeypatch):
    _patch_clone(monkeypatch)

    def runner(d: DebugState, *, registry) -> DebugState:
        d.stage = PipelineStage.DONE
        d.fix = FixResult(patch="diff", attempts=1, test_output="ok")
        return d

    _patch_stage(monkeypatch, runner)

    graph = gapp.build_graph()
    out = await graph.ainvoke({"messages": [HumanMessage(content="bug")]}, config=_cfg())
    assert out["debug"].stage == PipelineStage.DONE
    # The chat surface carries a human-readable summary.
    assert any("Fixed" in str(getattr(m, "content", "")) for m in out["messages"])


async def test_failure_routes_to_human_review_and_interrupts(monkeypatch):
    _patch_clone(monkeypatch)

    def runner(d: DebugState, *, registry) -> DebugState:
        d.stage = PipelineStage.FAILED
        d.error = "boom"
        return d

    _patch_stage(monkeypatch, runner)

    graph = gapp.build_graph()
    cfg = _cfg()
    out = await graph.ainvoke({"messages": [HumanMessage(content="bug")]}, config=cfg)
    # The run paused inside human_review awaiting feedback.
    assert "__interrupt__" in out
    snap = await graph.aget_state(cfg)
    assert "human_review" in snap.next


async def test_resume_injects_feedback_and_loops_back(monkeypatch):
    _patch_clone(monkeypatch)
    calls = {"n": 0}

    def runner(d: DebugState, *, registry) -> DebugState:
        calls["n"] += 1
        if calls["n"] == 1:
            d.stage = PipelineStage.FAILED
            d.error = "boom"
        else:
            # second attempt only succeeds because feedback arrived.
            assert d.user_feedback == "focus on parser.py"
            d.stage = PipelineStage.DONE
            d.fix = FixResult(patch="diff", attempts=2, test_output="ok")
        return d

    _patch_stage(monkeypatch, runner)

    graph = gapp.build_graph()
    cfg = _cfg()
    await graph.ainvoke({"messages": [HumanMessage(content="bug")]}, config=cfg)
    out = await graph.ainvoke(Command(resume="focus on parser.py"), config=cfg)
    assert out["debug"].stage == PipelineStage.DONE
    assert calls["n"] == 2


async def test_clone_failure_ends_without_human_review(monkeypatch):
    # Infra failure (Docker down) is not fixable by human guidance -> end, no HITL.
    def boom(state):
        raise RuntimeError("docker not running")

    monkeypatch.setattr(gapp, "clone_repo", boom)
    monkeypatch.setattr(gapp, "store_agent_run", lambda *a, **k: None)

    graph = gapp.build_graph()
    cfg = _cfg()
    out = await graph.ainvoke({"messages": [HumanMessage(content="bug")]}, config=cfg)
    assert out["debug"].stage == PipelineStage.FAILED
    assert "__interrupt__" not in out
    snap = await graph.aget_state(cfg)
    assert not snap.next  # terminated
