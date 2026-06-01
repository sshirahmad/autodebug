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


# ---------------------------------------------------------------------------
# run_root_cause
# ---------------------------------------------------------------------------

class TestRunRootCause:
    def test_advances_to_fix_when_result_populated(self, bisect_state, capture_result_list):
        def behavior():
            capture_result_list[-1].append(
                RootCauseResult(
                    summary="op bug",
                    relevant_lines=["calc.py:2"],
                    hypothesis="- was used instead of +",
                )
            )
        sb = _make_fake_sandbox()
        sb.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="AssertionError")
        with patch_sandbox("autodebug.agents.root_cause", sb), \
             patch_create_agent("autodebug.agents.root_cause", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.root_cause import run_root_cause
            state = run_root_cause(bisect_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FIX
        assert state.root_cause.hypothesis == "- was used instead of +"


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
