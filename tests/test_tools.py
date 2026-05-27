"""Unit tests for individual tool implementations."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from autodebug.tools.shared import make_read_file_tool, make_list_files_tool
from autodebug.tools.repro import make_run_script_tool, make_submit_repro_tool
from autodebug.tools.fix import make_apply_patch_tool


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_content(repo: Path):
    tool = make_read_file_tool(repo_path=repo)
    result = tool.invoke({"path": "calc.py"})
    assert "def add" in result


def test_read_file_missing(repo: Path):
    tool = make_read_file_tool(repo_path=repo)
    result = tool.invoke({"path": "nonexistent.py"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

def test_list_files_returns_entries(repo: Path):
    tool = make_list_files_tool(repo_path=repo)
    result = tool.invoke({"path": "."})
    assert "calc.py" in result


def test_list_files_missing_dir(repo: Path):
    tool = make_list_files_tool(repo_path=repo)
    result = tool.invoke({"path": "no_such_dir"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------

def test_run_script_returns_json():
    sandbox = MagicMock()
    sandbox.run_script.return_value = MagicMock(exit_code=0, stdout="ok", stderr="")
    tool = make_run_script_tool(sandbox=sandbox)
    result = tool.invoke({"script": "print('hi')"})
    data = json.loads(result)
    assert data["exit_code"] == 0


# ---------------------------------------------------------------------------
# submit_repro
# ---------------------------------------------------------------------------

def test_submit_repro_confirmed_when_script_fails():
    sandbox = MagicMock()
    sandbox.run_script.return_value = MagicMock(success=False, exit_code=1)
    result_holder = []
    tool = make_submit_repro_tool(sandbox=sandbox, result=result_holder)
    output = tool.invoke({"script": "raise ValueError()", "error_output": "ValueError"})
    assert "confirmed" in output.lower()
    assert len(result_holder) == 1
    assert result_holder[0].confirmed is True


def test_submit_repro_rejected_when_script_passes():
    sandbox = MagicMock()
    sandbox.run_script.return_value = MagicMock(success=True, exit_code=0)
    result_holder = []
    tool = make_submit_repro_tool(sandbox=sandbox, result=result_holder)
    output = tool.invoke({"script": "print('ok')", "error_output": ""})
    assert "not reproduced" in output.lower()
    assert len(result_holder) == 0


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

def test_apply_patch_success(repo: Path):
    patches = []
    tool = make_apply_patch_tool(repo_path=repo, patches=patches)
    result = tool.invoke({
        "path": "calc.py",
        "old_content": "return a - b  # bug",
        "new_content": "return a + b",
    })
    assert "applied" in result.lower()
    assert len(patches) == 1
    assert (repo / "calc.py").read_text().count("return a + b") == 1


def test_apply_patch_old_content_not_found(repo: Path):
    patches = []
    tool = make_apply_patch_tool(repo_path=repo, patches=patches)
    result = tool.invoke({
        "path": "calc.py",
        "old_content": "this does not exist in the file",
        "new_content": "something",
    })
    assert "error" in result.lower()
    assert len(patches) == 0


def test_apply_patch_file_not_found(repo: Path):
    tool = make_apply_patch_tool(repo_path=repo, patches=[])
    result = tool.invoke({"path": "ghost.py", "old_content": "x", "new_content": "y"})
    assert "not found" in result.lower()
