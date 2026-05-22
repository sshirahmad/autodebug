"""Unit tests for ReproAgent using a mock sandbox and mock LLM."""

import pytest
from unittest.mock import MagicMock, patch

from autodebug.agents.repro import ReproAgent
from autodebug.state import DebugState, PipelineStage


FAKE_SCRIPT = "raise ValueError('bug!')"
FAKE_ERROR = "ValueError: bug!"


def _make_tool_use_block(name: str, args: dict, block_id: str = "tu_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    block.id = block_id
    return block


def _make_response(blocks: list, input_tokens=100, output_tokens=200):
    response = MagicMock()
    response.content = blocks
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


@pytest.fixture
def state(tmp_path):
    # Create a minimal fake repo
    (tmp_path / "main.py").write_text("def add(a, b): return a - b  # bug: should be +\n")
    return DebugState(
        repo_url="https://github.com/fake/repo",
        bug_report="add(1, 2) returns -1 instead of 3",
        repo_local_path=str(tmp_path),
    )


def test_repro_agent_succeeds_on_first_attempt(state):
    agent = ReproAgent(client=MagicMock())

    # LLM immediately calls submit_repro with a valid script
    submit_block = _make_tool_use_block(
        "submit_repro",
        {"script": FAKE_SCRIPT, "error_output": FAKE_ERROR},
    )
    agent._chat = MagicMock(return_value=_make_response([submit_block]))

    # Sandbox confirms the script fails (exit_code != 0)
    with patch("autodebug.agents.repro.Sandbox") as MockSandbox:
        mock_sandbox = MockSandbox.return_value
        mock_sandbox.run_script.return_value = MagicMock(success=False, exit_code=1)

        result = agent.run(state)

    assert result.repro is not None
    assert result.repro.confirmed is True
    assert result.repro.repro_script == FAKE_SCRIPT
    assert result.stage == PipelineStage.BISECT


def test_repro_agent_fails_if_script_passes(state):
    agent = ReproAgent(client=MagicMock())

    submit_block = _make_tool_use_block(
        "submit_repro",
        {"script": "print('no error')", "error_output": ""},
    )
    agent._chat = MagicMock(return_value=_make_response([submit_block]))

    with patch("autodebug.agents.repro.Sandbox") as MockSandbox:
        mock_sandbox = MockSandbox.return_value
        # Script passes — bug not reproduced
        mock_sandbox.run_script.return_value = MagicMock(success=True, exit_code=0)

        result = agent.run(state)

    # Should NOT succeed — agent needs to keep trying but runs out of attempts
    assert result.stage == PipelineStage.FAILED


def test_repro_agent_counts_tokens(state):
    agent = ReproAgent(client=MagicMock())

    submit_block = _make_tool_use_block(
        "submit_repro",
        {"script": FAKE_SCRIPT, "error_output": FAKE_ERROR},
    )
    agent._chat = MagicMock(return_value=_make_response([submit_block], 150, 250))

    with patch("autodebug.agents.repro.Sandbox") as MockSandbox:
        mock_sandbox = MockSandbox.return_value
        mock_sandbox.run_script.return_value = MagicMock(success=False, exit_code=1)
        result = agent.run(state)

    assert result.total_tokens == 400
    assert result.total_llm_calls == 1
