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


def _patch_langmem_by_model(monkeypatch, behavior):
    """build_model passes the model_id straight through, and each optimizer's
    invoke() dispatches on it via `behavior(model_id) -> output|raises`. Lets a
    test fail on the agent model and succeed on the fallback."""
    import langmem

    monkeypatch.setattr(base, "build_model", lambda model_id=None, provider=None: model_id)

    class _Opt:
        def __init__(self, model):
            self.model = model

        def invoke(self, inp):
            return behavior(self.model)

    monkeypatch.setattr(langmem, "create_prompt_optimizer",
                        lambda model, kind="gradient": _Opt(model))


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

    def test_uses_the_dedicated_optimizer_model_not_the_agent_model(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        monkeypatch.setenv("AUTODEBUG_PROMPT_OPTIM_MODEL", "opt-model")
        used = {}

        def behavior(model):
            used["model"] = model
            return "IMPROVED"

        _patch_langmem_by_model(monkeypatch, behavior)
        out = base.maybe_optimize_prompt(
            "ORIG", [AIMessage(content="x")], "fb", model_id="agent-model")
        assert out == "IMPROVED"
        assert used["model"] == "opt-model"  # the dedicated model, not "agent-model"

    def test_defaults_to_gpt_4o_mini_when_unset(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM_MODEL", raising=False)
        used = {}

        def behavior(model):
            used["model"] = model
            return "IMPROVED"

        _patch_langmem_by_model(monkeypatch, behavior)
        base.maybe_optimize_prompt("ORIG", [AIMessage(content="x")], "fb", model_id="agent-model")
        assert used["model"] == base._DEFAULT_OPTIM_MODEL


class TestBraceMasking:
    """LangMem reads literal {x} as required f-string variables; our static
    prompts/trajectories carry code with braces (e.g. {sha}). We mask them for the
    optimizer round-trip so it doesn't demand spurious variables or crash .format()."""

    def test_mask_then_unmask_is_identity(self):
        s = 'use {sha}; idx = commits[len(commits)//2]; first_bad = {x}'
        masked = base._mask_braces(s)
        assert "{" not in masked and "}" not in masked
        assert base._unmask_braces(masked) == s

    def test_masked_prompt_exposes_no_fstring_variables(self):
        import re
        masked = base._mask_braces("find {sha} at {len(commits)} -- {first_bad}")
        assert re.findall(r"\{(.+?)\}", masked) == []  # LangMem extracts nothing

    def test_optimizer_receives_masked_prompt_and_result_is_unmasked(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_PROMPT_OPTIM", raising=False)
        seen = {}

        def behavior(_model):
            return "improved with {sha}"   # model echoes a brace -> must round-trip

        # capture what the optimizer was handed
        import langmem
        monkeypatch.setattr(base, "build_model", lambda model_id=None, provider=None: model_id)

        class _Opt:
            def __init__(self, model): pass
            def invoke(self, inp):
                seen["prompt"] = inp["prompt"]
                return behavior(None)

        monkeypatch.setattr(langmem, "create_prompt_optimizer",
                            lambda model, kind="gradient": _Opt(model))
        out = base.maybe_optimize_prompt("keep {sha} here", [AIMessage(content="x {y}")], "fb")
        assert "{" not in seen["prompt"] and "}" not in seen["prompt"]  # masked going in
        assert out == "improved with {sha}"                            # unmasked coming out


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
