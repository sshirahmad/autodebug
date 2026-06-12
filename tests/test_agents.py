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
    """Replacement for the CompiledStateGraph returned by create_agent."""

    def __init__(self, on_invoke):
        self._on_invoke = on_invoke

    def invoke(self, *args, **kwargs):
        return self._on_invoke()


def patch_create_agent(module_path: str, behavior):
    def fake_create_agent(**kwargs):
        return FakeAgent(behavior)
    return patch(f"{module_path}.create_agent", new=fake_create_agent)


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


@pytest.fixture
def capture_result_list(monkeypatch):
    """Capture the `result` list kwarg as each agent's tools are built."""
    holder: list = []

    from autodebug.registry import AutoDebugRegistry
    original = AutoDebugRegistry.build_tools

    def patched(self, agent_name, **ctx):
        if "result" in ctx:
            holder.append(ctx["result"])
        return original(self, agent_name, **ctx)

    monkeypatch.setattr(AutoDebugRegistry, "build_tools", patched)
    return holder


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
    def test_advances_to_bisect_when_result_populated(self, base_state, capture_result_list):
        def behavior():
            capture_result_list[0].append(
                ReproResult(repro_script="x", error_output="boom", confirmed=True)
            )
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.BISECT
        assert state.repro is not None
        assert state.repro.confirmed is True

    def test_fails_when_no_result_after_retries(self, base_state, capture_result_list):
        def behavior():
            pass
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "reproduce" in (state.error or "").lower()

    def test_fails_when_budget_exceeded_on_every_attempt(self, base_state, capture_result_list):
        def behavior():
            raise BudgetExceeded("token budget hit")
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.repro", sb), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "budget exceeded" in (state.error or "").lower()

    def test_degrades_to_failed_on_unexpected_error(self, base_state, capture_result_list):
        """A non-budget exception (transient provider/tool error) must degrade to a
        FAILED state with the cause recorded — not propagate and crash the pipeline."""
        def behavior():
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
# run_root_cause
# ---------------------------------------------------------------------------

class TestRunRootCause:
    def test_advances_to_fix_on_evidence_backed_report(self, bisect_state):
        # New mechanism: the agent returns a structured RootCauseReport AND must
        # have called an inspection tool — the driver enforces both.
        from langchain_core.messages import AIMessage
        from autodebug.state import RootCauseReport

        def behavior():
            return {
                "structured_response": RootCauseReport(
                    summary="op bug", hypothesis="- was used instead of +",
                    relevant_lines=["calc.py:2"], evidence="observed at calc.py:2",
                ),
                "messages": [AIMessage(content="", tool_calls=[
                    {"name": "inspect_at", "args": {}, "id": "1", "type": "tool_call"}])],
            }
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="AssertionError")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             patch_create_agent("autodebug.agents.root_cause", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(bisect_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FIX
        assert state.root_cause.hypothesis == "- was used instead of +"
        assert state.root_cause.evidence == "observed at calc.py:2"

    def test_rejected_when_report_lacks_observed_evidence(self, bisect_state):
        # A report with NO inspection tool call must be rejected (not advance).
        from autodebug.state import RootCauseReport

        def behavior():
            return {  # structured report but never called inspect_at/postmortem
                "structured_response": RootCauseReport(
                    summary="guess", hypothesis="speculation", relevant_lines=[], evidence="made up"),
                "messages": [],
            }
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="x")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             patch_create_agent("autodebug.agents.root_cause", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(bisect_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "without observing" in (state.error or "")


# ---------------------------------------------------------------------------
# run_fix
# ---------------------------------------------------------------------------

class TestRunFix:
    def test_advances_to_done_when_result_populated(self, root_cause_state, capture_result_list):
        def behavior():
            results = [lst for lst in capture_result_list if isinstance(lst, list)]
            results[-1].append(FixResult(patch="x", attempts=1, test_output="all passed"))
        sb = _make_fake_sandbox()
        with patch_sandbox("autodebug.agents.fix", sb), \
             patch_create_agent("autodebug.agents.fix", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.fix import run_fix
            state = run_fix(root_cause_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.DONE
        assert state.fix is not None


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
    def test_jumps_to_end_when_result_populated(self):
        from autodebug.agents.base import submission_middleware
        result = ["submitted"]
        mw = submission_middleware(result)
        assert len(mw) == 1
        # Call the underlying hook function directly (decorated as before_model)
        out = mw[0].before_model(state={"messages": []}, runtime=None)
        assert out == {"jump_to": "end"}

    def test_returns_none_when_result_empty(self):
        from autodebug.agents.base import submission_middleware
        result: list = []
        mw = submission_middleware(result)
        out = mw[0].before_model(state={"messages": []}, runtime=None)
        assert out is None

    def test_combined_with_budget_middleware(self):
        from autodebug.agents.base import Budget, budget_middleware, submission_middleware
        budget = Budget()
        combined = budget_middleware(budget) + submission_middleware([])
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
