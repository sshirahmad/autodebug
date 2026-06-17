"""Tests for agent runner functions (run_repro, run_bisect, run_root_cause, run_fix).

We mock `create_agent` so its `.invoke()` either populates the runner's `result`
list (success path) or raises `BudgetExceeded` (budget path) without ever touching
a real LLM. We also mock `Sandbox` so the runner's context-manager block returns
our fake without spinning up a real container.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from autodebug.agents.base import BudgetExceeded
from autodebug.sandbox import RunResult
from autodebug.state import (
    BisectResult,
    DebugState,
    FixResult,
    PipelineStage,
    ReproResult,
    RootCauseResult,
)


# ---------------------------------------------------------------------------
# Mocking helpers
# ---------------------------------------------------------------------------

class FakeAgent:
    """Replacement for the compiled agent. A submit is simulated by `behavior`
    writing into `self.state` (e.g. self.state["repro"] = {...}); the driver reads
    it back via get_state(), mirroring the real result-channel flow."""

    def __init__(self, behavior=None):
        self.state = {"messages": []}
        self._behavior = behavior

    def invoke(self, payload, **kwargs):
        if self._behavior:
            self._behavior(self)   # may write self.state[...] or raise
        return self.state

    def get_state(self, config):
        return type("S", (), {"values": dict(self.state)})()


def patch_create_agent(module_path: str, behavior=None):
    return patch(f"{module_path}.create_agent", new=lambda **kw: FakeAgent(behavior))


class _CapturingAgent:
    """Records the initial human-message content; satisfies get_state()."""
    def __init__(self, sink):
        self._sink = sink

    def invoke(self, payload, **kwargs):
        self._sink["text"] = payload["messages"][0].content
        return {}

    def get_state(self, config):
        return type("S", (), {"values": {"messages": []}})()


def patch_capturing_create_agent(module_path: str, sink: dict):
    return patch(f"{module_path}.create_agent", new=lambda **kw: _CapturingAgent(sink))


def _make_fake_sandbox() -> MagicMock:
    """Sandbox mock that supports the `with` protocol and the methods runners call."""
    sb = MagicMock()
    ok = RunResult(exit_code=0, stdout="", stderr="")
    sb.run_script.return_value = ok
    sb.exec.return_value = ok
    sb.git.return_value = ok
    sb.read_file.return_value = "(fake)"
    sb.list_files.return_value = ""
    sb.write_file.return_value = ok
    # context manager protocol — `with Sandbox(...) as s:` yields the same mock
    sb.__enter__.return_value = sb
    sb.__exit__.return_value = None
    return sb


def patch_sandbox(module_path: str, fake: MagicMock):
    """Patch `Sandbox(volume=...)` to return our fake instance in that module."""
    return patch(f"{module_path}.Sandbox", return_value=fake)


@pytest.fixture(autouse=True)
def _isolate_resume(monkeypatch, tmp_path):
    """Keep unit tests hermetic: no cross-run resume short-circuit (so the mocked
    agent actually runs) and a throwaway checkpoint dir."""
    monkeypatch.setenv("AUTODEBUG_RESUME", "0")
    monkeypatch.setenv("AUTODEBUG_CHECKPOINT_DIR", str(tmp_path))
    from autodebug import resume
    resume._saver = None
    yield
    resume._saver = None


@contextmanager
def _patched_bisect_helpers():
    """Mock git_utils calls bisect makes before agent.invoke (unshallow, current_sha, get_commit_info)."""
    from autodebug.tools.git_utils import CommitInfo
    info = CommitInfo(sha="deadbeef", short_sha="dead", message="initial",
                      author="x", date="2024-01-01", diff="")
    with patch("autodebug.agents.bisect.git_utils.unshallow") as un, \
         patch("autodebug.agents.bisect.git_utils.current_sha", return_value="deadbeef") as cs, \
         patch("autodebug.agents.bisect.git_utils.get_commit_info", return_value=info) as ci:
        yield un, cs, ci


# ---------------------------------------------------------------------------
# run_repro
# ---------------------------------------------------------------------------

class TestRunRepro:
    def test_advances_to_bisect_when_result_populated(self, base_state):
        def behavior(agent):
            agent.state["repro"] = ReproResult(
                repro_script="x", error_output="boom", confirmed=True).model_dump()
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.BISECT
        assert state.repro is not None
        assert state.repro.confirmed is True

    def test_fails_when_no_result_after_retries(self, base_state):
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior=None):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "reproduce" in (state.error or "").lower()

    def test_fails_when_budget_exceeded_on_every_attempt(self, base_state):
        def behavior(agent):
            raise BudgetExceeded("token budget hit")
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "budget exceeded" in (state.error or "").lower()

    def test_degrades_to_failed_on_unexpected_error(self, base_state):
        """A non-budget exception (transient provider/tool error) must degrade to a
        FAILED state with the cause recorded — not propagate and crash the pipeline."""
        def behavior(agent):
            raise RuntimeError("Provider returned error 400")
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "Provider returned error 400" in (state.error or "")


# ---------------------------------------------------------------------------
# run_bisect — must always hand the shared volume back at the original HEAD
# ---------------------------------------------------------------------------

class TestRunBisect:
    def test_restores_checkout_on_success(self, repro_state):
        def behavior(agent):
            agent.state["bisect"] = BisectResult(
                culprit_commit="c0ffee", commit_message="m", commit_diff="d",
                steps_taken=0).model_dump()
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.bisect", sb), \
             patch_create_agent("autodebug.agents.bisect", behavior), \
             _patched_bisect_helpers(), \
             patch("autodebug.agents.bisect.git_utils.restore_checkout") as restore:
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.bisect import run_bisect
            state = run_bisect(repro_state, registry=AutoDebugRegistry.from_file())
        assert state.stage == PipelineStage.ROOT_CAUSE
        restore.assert_called_once_with(sb, "deadbeef")  # original HEAD from helpers

    def test_advances_to_root_cause_without_culprit_and_restores(self, repro_state):
        # No submission: bisect is best-effort, so it ADVANCES to root_cause
        # (state.bisect=None) rather than failing — and still restores the volume.
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.bisect", sb), \
             patch_create_agent("autodebug.agents.bisect", behavior=None), \
             _patched_bisect_helpers(), \
             patch("autodebug.agents.bisect.git_utils.restore_checkout") as restore:
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.bisect import run_bisect
            state = run_bisect(repro_state, registry=AutoDebugRegistry.from_file())
        assert state.stage == PipelineStage.ROOT_CAUSE
        assert state.bisect is None
        restore.assert_called_once_with(sb, "deadbeef")


# ---------------------------------------------------------------------------
# run_root_cause
# ---------------------------------------------------------------------------

def _patch_root_cause_agent(rc, inspected: bool):
    """Fake create_agent: writes the root_cause channel and reports (via messages)
    whether an inspection tool was used."""
    from langchain_core.messages import AIMessage

    def behavior(agent):
        agent.state["root_cause"] = rc.model_dump()
        agent.state["messages"] = (
            [AIMessage(content="", tool_calls=[
                {"name": "inspect_at", "args": {}, "id": "1", "type": "tool_call"}])]
            if inspected else []
        )

    return patch("autodebug.agents.root_cause.create_agent", new=lambda **kw: FakeAgent(behavior))


class TestRunRootCause:
    def test_advances_to_fix_when_inspected_and_submitted(self, bisect_state):
        rc = RootCauseResult(
            summary="op bug", relevant_lines=["calc.py:2"],
            hypothesis="- used instead of +", evidence="observed at calc.py:2")
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="AssertionError")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             _patch_root_cause_agent(rc, inspected=True):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(bisect_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FIX
        assert state.root_cause.hypothesis == "- used instead of +"
        assert state.root_cause.evidence == "observed at calc.py:2"

    def test_does_not_hard_fail_without_inspection(self, bisect_state):
        # No inspection tool used: rejected on early attempts but ACCEPTED on the
        # final one — the pipeline degrades gracefully instead of failing outright
        # (the regression that response_format introduced).
        rc = RootCauseResult(summary="s", relevant_lines=["f:1"], hypothesis="h", evidence="e")
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="x")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             _patch_root_cause_agent(rc, inspected=False):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(bisect_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FIX  # graceful accept, not FAILED

    def test_runs_without_a_culprit(self, repro_state):
        # Bisect was inconclusive (state.bisect is None) — root_cause still runs
        # from the repro + bug report and advances to fix.
        assert repro_state.bisect is None
        rc = RootCauseResult(summary="s", relevant_lines=["f:1"], hypothesis="h", evidence="e")
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="x")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             _patch_root_cause_agent(rc, inspected=True):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(repro_state, registry=AutoDebugRegistry.from_file())
        assert state.stage == PipelineStage.FIX
        assert state.bisect is None


# ---------------------------------------------------------------------------
# run_fix
# ---------------------------------------------------------------------------

class TestRunFix:
    def test_advances_to_done_when_result_populated(self, root_cause_state):
        def behavior(agent):
            agent.state["fix"] = FixResult(
                patch="x", attempts=1, test_output="all passed").model_dump()
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.fix", sb), \
             patch_create_agent("autodebug.agents.fix", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.fix import run_fix
            state = run_fix(root_cause_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.DONE
        assert state.fix is not None

    def test_prompt_leads_with_the_fix_plan(self, root_cause_state):
        # Phase 2 handoff: the fixer is told to EXECUTE the root-cause plan.
        root_cause_state.root_cause.fix_plan = "Wrap the pool in try/except OSError at black.py:614"
        root_cause_state.root_cause.evidence = "OSError observed at black.py:632"
        sink = {}
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.fix", sb), \
             patch_capturing_create_agent("autodebug.agents.fix", sink):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.fix import run_fix
            run_fix(root_cause_state, registry=AutoDebugRegistry.from_file())
        text = sink["text"]
        assert "FIX PLAN" in text and "try/except OSError at black.py:614" in text
        assert "OSError observed at black.py:632" in text       # evidence forwarded
        assert "Do NOT re-investigate" in text


# ---------------------------------------------------------------------------
# Budget tracking via middleware (unit-level)
# ---------------------------------------------------------------------------

class TestBudgetMiddleware:
    def test_check_raises_when_cost_threshold_exceeded(self):
        from autodebug.agents.base import Budget, BudgetExceeded
        b = Budget(time_seconds=None, cost_usd=0.01,
                   cost_per_1k_input=0.003, cost_per_1k_output=0.015)
        b.add_tokens(1000, 200)
        b.check()
        b.add_tokens(500, 200)
        with pytest.raises(BudgetExceeded, match="Cost budget"):
            b.check()

    def test_cost_tracking(self):
        from autodebug.agents.base import Budget
        b = Budget(cost_per_1k_input=0.003, cost_per_1k_output=0.015)
        cost = b.add_tokens(1000, 500)
        assert abs(cost - 0.0105) < 1e-9
        assert abs(b.cost_used - 0.0105) < 1e-9

    def test_middleware_returns_pair(self):
        from autodebug.agents.base import Budget, budget_middleware
        mw = budget_middleware(Budget())
        assert len(mw) == 2


class TestReviseRefinement:
    """When re-invoked with a prior artifact (manager looped back from a failed
    fix), repro/root_cause refine the previous result instead of regenerating."""

    def test_repro_includes_prior_script_on_rerun(self):
        from autodebug.registry import AutoDebugRegistry
        from autodebug.state import DebugState, ReproResult
        sink: dict = {}
        st = DebugState(repo_url="u", bug_report="THE BUG", repo_volume="v")
        st.repro = ReproResult(repro_script="PRIOR_SCRIPT", error_output="PRIOR_FAIL", confirmed=True)
        with patch_capturing_create_agent("autodebug.agents.repro", sink), \
             patch_sandbox("autodebug.agents.repro", _make_fake_sandbox()):
            from autodebug.agents.repro import run_repro
            run_repro(st, registry=AutoDebugRegistry.from_file())
        assert "PRIOR_SCRIPT" in sink["text"]
        assert "Refine THIS reproduction" in sink["text"]

    def test_repro_has_no_refine_note_when_fresh(self):
        from autodebug.registry import AutoDebugRegistry
        from autodebug.state import DebugState
        sink: dict = {}
        st = DebugState(repo_url="u", bug_report="THE BUG", repo_volume="v")
        with patch_capturing_create_agent("autodebug.agents.repro", sink), \
             patch_sandbox("autodebug.agents.repro", _make_fake_sandbox()):
            from autodebug.agents.repro import run_repro
            run_repro(st, registry=AutoDebugRegistry.from_file())
        assert "Refine THIS reproduction" not in sink["text"]

    def test_root_cause_includes_prior_hypothesis_on_rerun(self):
        from autodebug.registry import AutoDebugRegistry
        from autodebug.state import DebugState, ReproResult, BisectResult, RootCauseResult
        sink: dict = {}
        st = DebugState(repo_url="u", bug_report="b", repo_volume="v")
        st.repro = ReproResult(repro_script="s", error_output="e", confirmed=True)
        st.bisect = BisectResult(culprit_commit="abc", commit_message="m", commit_diff="d", steps_taken=1)
        st.root_cause = RootCauseResult(summary="S", relevant_lines=[], hypothesis="PRIOR_HYPOTHESIS")
        with patch_capturing_create_agent("autodebug.agents.root_cause", sink), \
             patch_sandbox("autodebug.agents.root_cause", _make_fake_sandbox()):
            from autodebug.agents.root_cause import run_root_cause
            run_root_cause(st, registry=AutoDebugRegistry.from_file())
        assert "PRIOR_HYPOTHESIS" in sink["text"]
        assert "got wrong" in sink["text"]


class TestSessionBudgetMiddleware:
    """Global ceiling across the whole orchestrated session (manager + sub-agents)."""

    def _hook(self, state, max_cost=None, max_seconds=None):
        from autodebug.agents.base import session_budget_middleware
        return session_budget_middleware(state, max_cost, max_seconds)[0]

    def test_passes_under_cost_cap(self):
        from autodebug.state import DebugState
        st = DebugState(repo_url="u", bug_report="b")
        st.total_cost = 5.0
        assert self._hook(st, max_cost=8.0).before_model(state={"messages": []}, runtime=None) is None

    def test_raises_over_cost_cap(self):
        from autodebug.agents.base import BudgetExceeded
        from autodebug.state import DebugState
        st = DebugState(repo_url="u", bug_report="b")
        st.total_cost = 9.5
        with pytest.raises(BudgetExceeded, match="Session cost budget"):
            self._hook(st, max_cost=8.0).before_model(state={"messages": []}, runtime=None)

    def test_no_caps_never_raises(self):
        from autodebug.state import DebugState
        st = DebugState(repo_url="u", bug_report="b")
        st.total_cost = 9999
        assert self._hook(st).before_model(state={"messages": []}, runtime=None) is None

    def test_cap_reads_cumulative_state_cost_live(self):
        # The hook closes over the DebugState, so later sub-agent spend is seen.
        from autodebug.agents.base import BudgetExceeded
        from autodebug.state import DebugState
        st = DebugState(repo_url="u", bug_report="b")
        hook = self._hook(st, max_cost=2.0)
        st.total_cost = 1.0
        assert hook.before_model(state={"messages": []}, runtime=None) is None
        st.total_cost = 2.5  # a sub-agent finished and pushed cost over the cap
        with pytest.raises(BudgetExceeded):
            hook.before_model(state={"messages": []}, runtime=None)


class TestSubmissionMiddleware:
    def test_jumps_to_end_when_channel_populated(self):
        from autodebug.agents.base import submission_middleware
        mw = submission_middleware("repro")
        assert len(mw) == 1
        # Call the underlying hook directly: the result channel is set -> jump to end.
        out = mw[0].before_model(state={"messages": [], "repro": {"confirmed": True}}, runtime=None)
        assert out == {"jump_to": "end"}

    def test_returns_none_when_channel_empty(self):
        from autodebug.agents.base import submission_middleware
        mw = submission_middleware("repro")
        out = mw[0].before_model(state={"messages": [], "repro": None}, runtime=None)
        assert out is None

    def test_combined_with_budget_middleware(self):
        from autodebug.agents.base import Budget, budget_middleware, submission_middleware
        budget = Budget()
        combined = budget_middleware(budget) + submission_middleware("repro")
        assert len(combined) == 3  # check_budget + track_tokens + check_submitted


class TestRequireToolCallsMiddleware:
    def test_jumps_back_to_model_when_no_tool_calls(self):
        from langchain_core.messages import AIMessage
        from autodebug.agents.base import require_tool_calls_middleware

        mw = require_tool_calls_middleware()
        assert len(mw) == 1

        ai_msg = AIMessage(content="Some analysis text", tool_calls=[])
        out = mw[0].after_model(
            state={"messages": [ai_msg]}, runtime=None,
        )
        assert out is not None
        assert out["jump_to"] == "model"
        appended = out["messages"][0]
        assert "tool call" in appended.content.lower()

    def test_passes_through_when_tool_calls_present(self):
        from langchain_core.messages import AIMessage
        from autodebug.agents.base import require_tool_calls_middleware

        mw = require_tool_calls_middleware()
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1", "type": "tool_call"}],
        )
        out = mw[0].after_model(
            state={"messages": [ai_msg]}, runtime=None,
        )
        assert out is None

    def test_passes_through_when_last_message_is_not_ai(self):
        from langchain_core.messages import HumanMessage
        from autodebug.agents.base import require_tool_calls_middleware

        mw = require_tool_calls_middleware()
        out = mw[0].after_model(
            state={"messages": [HumanMessage(content="hi")]}, runtime=None,
        )
        assert out is None


# ---------------------------------------------------------------------------
# model_for_attempt — optionally escalate to a stronger model on retries
# ---------------------------------------------------------------------------

class TestModelForAttempt:
    def _capture(self, monkeypatch):
        from autodebug.agents import base
        seen = {}
        monkeypatch.setattr(
            base, "build_model",
            lambda model_id=None, provider=None: seen.update(model=model_id, provider=provider),
        )
        return base, seen

    def test_first_attempt_uses_base_model(self, monkeypatch):
        base, seen = self._capture(monkeypatch)
        monkeypatch.setenv("AUTODEBUG_RETRY_MODEL", "strong")  # set but attempt 0 ignores it
        base.model_for_attempt(0, "base", "prov")
        assert seen == {"model": "base", "provider": "prov"}

    def test_retry_escalates_when_retry_model_set(self, monkeypatch):
        base, seen = self._capture(monkeypatch)
        monkeypatch.setenv("AUTODEBUG_RETRY_MODEL", "strong")
        monkeypatch.delenv("AUTODEBUG_RETRY_MODEL_PROVIDER", raising=False)
        base.model_for_attempt(1, "base", "prov")
        assert seen == {"model": "strong", "provider": "prov"}  # provider falls back to base

    def test_retry_keeps_base_when_no_retry_model(self, monkeypatch):
        base, seen = self._capture(monkeypatch)
        monkeypatch.delenv("AUTODEBUG_RETRY_MODEL", raising=False)
        base.model_for_attempt(2, "base", "prov")
        assert seen == {"model": "base", "provider": "prov"}
