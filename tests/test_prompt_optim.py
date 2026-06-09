"""Tests for LangMem-based prompt optimization on agent retries.

The optimizer itself makes an LLM call, so we monkeypatch `langmem` and
`build_model` — these tests verify the *wiring*: env kill-switch, empty-history
short-circuit, the input we hand LangMem, trajectory trimming, checkpoint
recovery, and the never-break fallback on error.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.agents import base  # noqa: E402


class _FakeOptimizer:
    def __init__(self, out, sink):
        self._out, self._sink = out, sink

    def invoke(self, inp):
        self._sink["input"] = inp
        return self._out


def _patch_langmem(monkeypatch, out="IMPROVED PROMPT"):
    sink: dict = {}
    import langmem
    monkeypatch.setattr(base, "build_model", lambda **kw: object())
    monkeypatch.setattr(
        langmem, "create_prompt_optimizer",
        lambda model, kind="gradient": _FakeOptimizer(out, sink),
    )
    return sink


class TestMaybeOptimizePrompt:
    def test_disabled_via_env_returns_original(self, monkeypatch):
        monkeypatch.setenv("AUTODEBUG_PROMPT_OPTIM", "0")
        _patch_langmem(monkeypatch)
        assert base.maybe_optimize_prompt("P", [AIMessage(content="x")], "fb") == "P"

    def test_empty_history_returns_original(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        assert base.maybe_optimize_prompt("P", [], "fb") == "P"

    def test_optimizes_and_passes_trajectory_and_prompt(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        sink = _patch_langmem(monkeypatch, out="IMPROVED")
        msgs = [HumanMessage(content="bug"), AIMessage(content="attempt")]
        out = base.maybe_optimize_prompt("ORIGINAL", msgs, "you failed")
        assert out == "IMPROVED"
        assert sink["input"]["prompt"] == "ORIGINAL"
        traj = sink["input"]["trajectories"][0]
        assert traj.feedback == "you failed"
        assert traj.messages == msgs

    def test_optimizer_error_falls_back_to_original(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        import langmem
        monkeypatch.setattr(base, "build_model", lambda **kw: object())

        def boom(*a, **k):
            raise RuntimeError("optimizer down")

        monkeypatch.setattr(langmem, "create_prompt_optimizer", boom)
        assert base.maybe_optimize_prompt("ORIG", [AIMessage(content="x")], "fb") == "ORIG"

    def test_blank_optimizer_output_falls_back(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        _patch_langmem(monkeypatch, out="   ")
        assert base.maybe_optimize_prompt("ORIG", [AIMessage(content="x")], "fb") == "ORIG"


class TestTrimTrajectory:
    def test_keeps_short_trajectories_intact(self):
        msgs = [AIMessage(content=str(i)) for i in range(10)]
        assert base._trim_trajectory(msgs) == msgs

    def test_caps_long_trajectories_head_plus_tail(self):
        msgs = [AIMessage(content=str(i)) for i in range(200)]
        trimmed = base._trim_trajectory(msgs)
        assert len(trimmed) == base._TRAJ_HEAD + base._TRAJ_TAIL
        assert trimmed[0].content == "0"            # opening task framing kept
        assert trimmed[-1].content == "199"          # most recent turns kept


class TestAttemptTrajectory:
    def test_returns_messages_from_checkpoint(self):
        class Snap:
            values = {"messages": [1, 2, 3]}

        class Agent:
            def get_state(self, cfg):
                return Snap()

        assert base.attempt_trajectory(Agent(), {}) == [1, 2, 3]

    def test_returns_empty_on_error(self):
        class Agent:
            def get_state(self, cfg):
                raise RuntimeError("no checkpointer")

        assert base.attempt_trajectory(Agent(), {}) == []

    def test_returns_empty_when_get_state_missing(self):
        assert base.attempt_trajectory(object(), {}) == []


def test_retry_feedback_mentions_goal_and_failure():
    fb = base.retry_feedback("calling submit_repro")
    assert "submit_repro" in fb and "FAILED" in fb
