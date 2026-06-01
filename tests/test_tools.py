"""Unit tests for individual tool implementations.

Tools now route through the sandbox abstraction; we mock sandbox methods at
the level each tool uses (read_file / list_files / exec / write_file / run_script).
"""

import json
from unittest.mock import MagicMock

from autodebug.sandbox import RunResult
from autodebug.tools.shared import make_read_file_tool, make_list_files_tool
from autodebug.tools.repro import make_run_script_tool, make_submit_repro_tool
from autodebug.tools.fix import make_apply_patch_tool


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_content():
    sandbox = MagicMock()
    sandbox.read_file.return_value = "def add(a, b):\n    return a - b\n"
    tool = make_read_file_tool(sandbox=sandbox)
    result = tool.invoke({"path": "calc.py"})
    assert "def add" in result
    sandbox.read_file.assert_called_once()


def test_read_file_missing():
    sandbox = MagicMock()
    sandbox.read_file.return_value = "File not found: nonexistent.py"
    tool = make_read_file_tool(sandbox=sandbox)
    result = tool.invoke({"path": "nonexistent.py"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

def test_list_files_returns_entries():
    sandbox = MagicMock()
    sandbox.list_files.return_value = "calc.py\ntest_calc.py"
    tool = make_list_files_tool(sandbox=sandbox)
    result = tool.invoke({"path": "."})
    assert "calc.py" in result


def test_list_files_missing_dir():
    sandbox = MagicMock()
    sandbox.list_files.return_value = "Directory not found: no_such_dir"
    tool = make_list_files_tool(sandbox=sandbox)
    result = tool.invoke({"path": "no_such_dir"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------

def test_run_script_returns_json():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="ok", stderr="")
    tool = make_run_script_tool(sandbox=sandbox)
    result = tool.invoke({"script": "print('hi')"})
    data = json.loads(result)
    assert data["exit_code"] == 0


# ---------------------------------------------------------------------------
# submit_repro
# ---------------------------------------------------------------------------

def test_submit_repro_confirmed_when_script_fails():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="ValueError")
    result_holder = []
    tool = make_submit_repro_tool(sandbox=sandbox, result=result_holder)
    output = tool.invoke({"script": "raise ValueError()", "error_output": "ValueError"})
    assert "confirmed" in output.lower()
    assert len(result_holder) == 1
    assert result_holder[0].confirmed is True


def test_submit_repro_rejected_when_script_passes():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="", stderr="")
    result_holder = []
    tool = make_submit_repro_tool(sandbox=sandbox, result=result_holder)
    output = tool.invoke({"script": "print('ok')", "error_output": ""})
    assert "not reproduced" in output.lower()
    assert len(result_holder) == 0


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

def _sandbox_with_file(content: str) -> MagicMock:
    """Sandbox mock whose exec(cat ...) returns the given file content."""
    sandbox = MagicMock()
    # test -f succeeds, cat returns content
    def exec_side_effect(cmd, workdir=None):
        if cmd.startswith("test -f"):
            return RunResult(exit_code=0, stdout="", stderr="")
        if cmd.startswith("cat "):
            return RunResult(exit_code=0, stdout=content, stderr="")
        return RunResult(exit_code=0, stdout="", stderr="")
    sandbox.exec.side_effect = exec_side_effect
    sandbox.write_file.return_value = RunResult(exit_code=0, stdout="", stderr="")
    return sandbox


def test_apply_patch_success():
    patches = []
    sandbox = _sandbox_with_file("def add(a, b):\n    return a - b  # bug\n")
    tool = make_apply_patch_tool(sandbox=sandbox, patches=patches)
    result = tool.invoke({
        "path": "calc.py",
        "old_content": "return a - b  # bug",
        "new_content": "return a + b",
    })
    assert "applied" in result.lower()
    assert len(patches) == 1
    sandbox.write_file.assert_called_once()


def test_apply_patch_old_content_not_found():
    patches = []
    sandbox = _sandbox_with_file("def add(a, b):\n    return a - b  # bug\n")
    tool = make_apply_patch_tool(sandbox=sandbox, patches=patches)
    result = tool.invoke({
        "path": "calc.py",
        "old_content": "this does not exist in the file",
        "new_content": "something",
    })
    assert "error" in result.lower()
    assert len(patches) == 0


def test_apply_patch_file_not_found():
    sandbox = MagicMock()
    sandbox.exec.return_value = RunResult(exit_code=1, stdout="", stderr="")  # test -f fails
    tool = make_apply_patch_tool(sandbox=sandbox, patches=[])
    result = tool.invoke({"path": "ghost.py", "old_content": "x", "new_content": "y"})
    assert "not found" in result.lower()
