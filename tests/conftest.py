"""Shared fixtures for all tests."""

import pytest
from pathlib import Path
from langchain_core.messages import AIMessage

from autodebug.state import DebugState, PipelineStage, ReproResult, BisectResult, RootCauseResult


def ai_with_tool_call(name: str, args: dict, id: str = "tc1") -> AIMessage:
    """Return an AIMessage that contains a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )


def ai_text(text: str = "done") -> AIMessage:
    """Return an AIMessage with no tool calls (stops the agent loop)."""
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Minimal fake repo with one buggy file."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\ndef test_add(): assert add(1, 2) == 3\n"
    )
    return tmp_path


@pytest.fixture
def fake_sandbox():
    """A MagicMock that mimics the Sandbox API just enough for tests.

    All sandbox methods return successful RunResults so submit_* tools that
    re-run a script during validation accept the result.
    """
    from unittest.mock import MagicMock
    from autodebug.sandbox import RunResult

    sb = MagicMock()
    ok = RunResult(exit_code=0, stdout="", stderr="")
    sb.run_script.return_value = ok
    sb.exec.return_value = ok
    sb.git.return_value = ok
    sb.read_file.return_value = "(fake file)"
    sb.list_files.return_value = "calc.py"
    sb.write_file.return_value = ok
    return sb


@pytest.fixture
def base_state(repo: Path) -> DebugState:
    return DebugState(
        repo_url="https://github.com/fake/repo",
        bug_report="add(1, 2) returns -1 instead of 3",
        repo_volume="autodebug_test_volume",
    )


@pytest.fixture
def repro_state(base_state: DebugState) -> DebugState:
    base_state.repro = ReproResult(
        repro_script="from calc import add; assert add(1,2)==3",
        error_output="AssertionError",
        confirmed=True,
    )
    base_state.stage = PipelineStage.BISECT
    return base_state


@pytest.fixture
def bisect_state(repro_state: DebugState) -> DebugState:
    repro_state.bisect = BisectResult(
        culprit_commit="abc1234",
        commit_message="refactor: switch subtraction",
        commit_diff="-    return a + b\n+    return a - b",
        steps_taken=3,
    )
    repro_state.stage = PipelineStage.ROOT_CAUSE
    return repro_state


@pytest.fixture
def root_cause_state(bisect_state: DebugState) -> DebugState:
    bisect_state.root_cause = RootCauseResult(
        summary="Operator changed from + to -",
        relevant_lines=["calc.py:2"],
        hypothesis="Bug caused by changing + to - in add()",
    )
    bisect_state.stage = PipelineStage.FIX
    return bisect_state
