"""Tests for agent run() loops with a mocked LLM."""

import pytest
from unittest.mock import MagicMock, patch

from autodebug.agents.repro import ReproAgent
from autodebug.agents.root_cause import RootCauseAgent
from autodebug.agents.fix import FixAgent
from autodebug.state import DebugState, PipelineStage
from tests.conftest import ai_with_tool_call, ai_text


def _mock_sandbox(success=False, exit_code=1, stdout="err", stderr=""):
    sandbox = MagicMock()
    sandbox.run_script.return_value = MagicMock(
        success=success, exit_code=exit_code,
        stdout=stdout, stderr=stderr,
        output=stderr or stdout,
    )
    sandbox.run.return_value = MagicMock(
        success=True, exit_code=0, stdout="", stderr="", output=""
    )
    return sandbox


def _tool_builder(sandbox):
    from autodebug.tools import build_tools

    def build(result, patches=None, **ctx):
        return build_tools(
            list(ctx.pop("_names", [])),
            sandbox=sandbox, result=result,
            patches=patches or [],
            **ctx,
        )
    return build


# ---------------------------------------------------------------------------
# ReproAgent
# ---------------------------------------------------------------------------

class TestReproAgent:
    def _make_agent(self, responses, sandbox):
        def tool_builder(**ctx):
            from autodebug.tools import build_tools
            return build_tools(
                ["read_file", "list_files", "run_script", "submit_repro"],
                **ctx,
            )
        agent = ReproAgent(tool_builder=tool_builder)
        agent._chat = MagicMock(side_effect=responses)
        agent.llm = MagicMock()
        return agent

    def test_succeeds_on_first_submit(self, base_state, repo):
        sandbox = _mock_sandbox(success=False, exit_code=1)
        response = ai_with_tool_call("submit_repro", {
            "script": "raise ValueError()",
            "error_output": "ValueError",
        })
        agent = self._make_agent([response], sandbox)

        with patch("autodebug.agents.repro.Sandbox", return_value=sandbox):
            result = agent.run(base_state)

        assert result.stage == PipelineStage.BISECT
        assert result.repro.confirmed is True
        assert result.total_llm_calls == 1
        assert result.total_tokens == 150

    def test_retries_when_script_passes(self, base_state, repo):
        sandbox = _mock_sandbox(success=True, exit_code=0)
        bad_submit = ai_with_tool_call("submit_repro", {
            "script": "print('ok')", "error_output": ""
        })
        agent = self._make_agent([bad_submit, ai_text("giving up")], sandbox)

        with patch("autodebug.agents.repro.Sandbox", return_value=sandbox):
            result = agent.run(base_state)

        assert result.stage == PipelineStage.FAILED
        assert result.total_llm_calls == 2

    def test_stops_when_no_tool_calls(self, base_state):
        sandbox = _mock_sandbox()
        agent = self._make_agent([ai_text("I give up")], sandbox)

        with patch("autodebug.agents.repro.Sandbox", return_value=sandbox):
            result = agent.run(base_state)

        assert result.stage == PipelineStage.FAILED
        assert result.total_llm_calls == 1


# ---------------------------------------------------------------------------
# RootCauseAgent
# ---------------------------------------------------------------------------

class TestRootCauseAgent:
    def _make_agent(self, responses, sandbox):
        def tool_builder(**ctx):
            from autodebug.tools import build_tools
            return build_tools(
                ["read_file", "run_repro_with_traceback", "submit_root_cause"],
                **ctx,
            )
        agent = RootCauseAgent(tool_builder=tool_builder)
        agent._chat = MagicMock(side_effect=responses)
        agent.llm = MagicMock()
        return agent

    def test_submits_root_cause(self, bisect_state):
        sandbox = _mock_sandbox(success=False, exit_code=1, stdout="", stderr="AssertionError")
        response = ai_with_tool_call("submit_root_cause", {
            "summary": "Operator bug",
            "relevant_lines": ["calc.py:2"],
            "hypothesis": "- was used instead of +",
        })
        agent = self._make_agent([response], sandbox)

        with patch("autodebug.agents.root_cause.Sandbox", return_value=sandbox):
            result = agent.run(bisect_state)

        assert result.stage == PipelineStage.FIX
        assert result.root_cause.hypothesis == "- was used instead of +"
        assert result.total_llm_calls == 1


# ---------------------------------------------------------------------------
# FixAgent
# ---------------------------------------------------------------------------

class TestFixAgent:
    def _make_agent(self, responses, sandbox):
        def tool_builder(**ctx):
            from autodebug.tools import build_tools
            return build_tools(
                ["read_file", "apply_patch", "run_repro", "run_tests", "submit_fix"],
                **ctx,
            )
        agent = FixAgent(tool_builder=tool_builder)
        agent._chat = MagicMock(side_effect=responses)
        agent.llm = MagicMock()
        return agent

    def test_submits_fix_when_both_pass(self, root_cause_state, repo):
        sandbox = MagicMock()
        sandbox.run_script.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="", output=""
        )
        sandbox.run.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="", output=""
        )

        response = ai_with_tool_call("submit_fix", {"summary": "Changed - to +"})
        agent = self._make_agent([response], sandbox)

        with patch("autodebug.agents.fix.Sandbox", return_value=sandbox):
            result = agent.run(root_cause_state)

        assert result.stage == PipelineStage.DONE
        assert result.fix is not None

    def test_cannot_submit_when_repro_still_fails(self, root_cause_state, repo):
        sandbox = MagicMock()
        sandbox.run_script.return_value = MagicMock(
            success=False, exit_code=1, stdout="", stderr="still failing", output="still failing"
        )
        sandbox.run.return_value = MagicMock(
            success=True, exit_code=0, stdout="", stderr="", output=""
        )

        responses = [
            ai_with_tool_call("submit_fix", {"summary": "attempted fix"}),
            ai_text("giving up"),
        ]
        agent = self._make_agent(responses, sandbox)

        with patch("autodebug.agents.fix.Sandbox", return_value=sandbox):
            result = agent.run(root_cause_state)

        assert result.stage == PipelineStage.FAILED
