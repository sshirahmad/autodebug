"""Tests for agent runner functions (run_repro, run_bisect, run_root_cause, run_fix).

We mock `create_agent` so its `.invoke()` either populates the runner's `result`
list (success path) or raises `BudgetExceeded` (budget path) without ever touching
a real LLM. The runner itself is fully exercised end-to-end.
"""

from unittest.mock import MagicMock, patch

import pytest

from autodebug.agents.base import BudgetExceeded
from autodebug.state import (
    BisectResult,
    DebugState,
    FixResult,
    PipelineStage,
    ReproResult,
    RootCauseResult,
)


# ---------------------------------------------------------------------------
# create_agent mocking helpers
# ---------------------------------------------------------------------------

class FakeAgent:
    """Replacement for the CompiledStateGraph returned by create_agent."""

    def __init__(self, on_invoke):
        self._on_invoke = on_invoke

    def invoke(self, *args, **kwargs):
        return self._on_invoke()


def patch_create_agent(module_path: str, behavior):
    """Patch `create_agent` in the given runner module with a fake that calls
    `behavior()` on each `.invoke()`. `behavior` runs in the closure of the
    runner, so it can append to result lists captured via patch_build_tools.
    """
    def fake_create_agent(**kwargs):
        return FakeAgent(behavior)
    return patch(f"{module_path}.create_agent", new=fake_create_agent)


@pytest.fixture
def capture_result_list(monkeypatch):
    """Patch AutoDebugRegistry.build_tools to capture the `result` list kwarg.

    Returns a one-element holder; after run_*() is called, holder[0] is the
    list the submit_* tool would append to.
    """
    holder: list = []

    from autodebug.registry import AutoDebugRegistry
    original = AutoDebugRegistry.build_tools

    def patched(self, agent_name, **ctx):
        if "result" in ctx:
            holder.append(ctx["result"])
        return original(self, agent_name, **ctx)

    monkeypatch.setattr(AutoDebugRegistry, "build_tools", patched)
    return holder


# ---------------------------------------------------------------------------
# run_repro
# ---------------------------------------------------------------------------

class TestRunRepro:
    def test_advances_to_bisect_when_result_populated(self, base_state, capture_result_list):
        def behavior():
            capture_result_list[0].append(
                ReproResult(repro_script="x", error_output="boom", confirmed=True)
            )

        from autodebug.agents import repro as repro_mod
        with patch.object(repro_mod, "Sandbox", return_value=MagicMock()), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.BISECT
        assert state.repro is not None
        assert state.repro.confirmed is True

    def test_fails_when_no_result_after_retries(self, base_state, capture_result_list):
        def behavior():
            pass  # never populate result

        from autodebug.agents import repro as repro_mod
        with patch.object(repro_mod, "Sandbox", return_value=MagicMock()), \
             patch_create_agent("autodebug.agents.repro", behavior):
            from autodebug.registry import AutoDebugRegistry
            from autodebug.agents.repro import run_repro
            state = run_repro(base_state, registry=AutoDebugRegistry.from_file())

        assert state.stage == PipelineStage.FAILED
        assert "reproduce" in (state.error or "").lower()

    def test_fails_when_budget_exceeded_on_every_attempt(self, base_state, capture_result_list):
        def behavior():
            raise BudgetExceeded("token budget hit")

        from autodebug.agents import repro as repro_mod
        with patch.object(repro_mod, "Sandbox", return_value=MagicMock()), \
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

        sandbox = MagicMock()
        sandbox.run_script.return_value = MagicMock(output="AssertionError")
        from autodebug.agents import root_cause as rc_mod
        with patch.object(rc_mod, "Sandbox", return_value=sandbox), \
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
            # capture_result_list[-1] is the most recent result list captured;
            # for fix agent, patches list is captured separately — we only need result.
            results = [lst for lst in capture_result_list if isinstance(lst, list)]
            results[-1].append(
                FixResult(patch="x", attempts=1, test_output="all passed")
            )

        sandbox = MagicMock()
        sandbox.run_script.return_value = MagicMock(success=True, exit_code=0)
        sandbox.run.return_value = MagicMock(success=True, exit_code=0)
        from autodebug.agents import fix as fix_mod
        with patch.object(fix_mod, "Sandbox", return_value=sandbox), \
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
    def test_check_raises_after_threshold(self):
        from autodebug.agents.base import Budget, BudgetExceeded
        b = Budget(time_seconds=None, tokens=100, cost_usd=None)
        b.add_tokens(50, 30)
        b.check()  # 80 <= 100 -> ok
        b.add_tokens(20, 5)
        with pytest.raises(BudgetExceeded, match="Token budget"):
            b.check()

    def test_cost_tracking(self):
        from autodebug.agents.base import Budget
        b = Budget(cost_per_1k_input=0.003, cost_per_1k_output=0.015)
        cost = b.add_tokens(1000, 500)
        # 1000 * 0.003/1000 + 500 * 0.015/1000 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-9
        assert abs(b.cost_used - 0.0105) < 1e-9

    def test_middleware_returns_pair(self):
        from autodebug.agents.base import Budget, budget_middleware
        mw = budget_middleware(Budget())
        assert len(mw) == 2
