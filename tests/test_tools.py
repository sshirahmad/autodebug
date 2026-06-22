"""Unit tests for individual tool implementations.

Tools now route through the sandbox abstraction; we mock sandbox methods at
the level each tool uses (read_file / list_files / exec / write_file / run_script).
"""

import json
from unittest.mock import MagicMock

import pytest

from autodebug.sandbox import RunResult
from autodebug.tools.shared import make_read_file_tool, make_list_files_tool, make_shell_tool
from autodebug.tools.repro import make_run_script_tool, make_submit_repro_tool
from autodebug.tools.fix import make_apply_patch_tool, make_submit_fix_tool
from autodebug.tools.root_cause import make_submit_root_cause_tool


# ---------------------------------------------------------------------------
# Tool metadata visibility (descriptions + per-argument descriptions)
# ---------------------------------------------------------------------------

def _build_all_tools():
    """Instantiate every registered tool factory with mocked context."""
    from autodebug.registry import _TOOL_FACTORIES
    from autodebug.fsm import FSM
    ctx = dict(sandbox=MagicMock(), result=[], patches=[], verdict=[], info=MagicMock(),
               sha="x", parent_sha="x", repro_script="x", test_command="x",
               state=MagicMock(), registry=MagicMock(), fsm=FSM(), agent_name="root_cause")
    return {name: factory(**ctx) for name, factory in _TOOL_FACTORIES.items()}


def test_every_tool_has_a_description():
    for name, t in _build_all_tools().items():
        assert t.description and t.description.strip(), f"{name} has no description"


def test_every_tool_argument_has_a_description():
    """Guards against adding a tool without parse_docstring=True / Args: — the
    model can't see arg descriptions otherwise. Uses the model-facing schema
    (tool_call_schema), which excludes injected args like tool_call_id."""
    missing = {}
    for name, t in _build_all_tools().items():
        schema = getattr(t, "tool_call_schema", None) or t.args_schema
        if not schema:
            continue
        props = schema.model_json_schema().get("properties", {})
        undocumented = [a for a, v in props.items() if not v.get("description")]
        if undocumented:
            missing[name] = undocumented
    assert not missing, f"tools with undocumented args (add parse_docstring + Args:): {missing}"


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

def _call(tool, **args):
    """Invoke a tool via a tool-call so injected args (tool_call_id) are filled.
    Returns a Command (accepted submit) or a ToolMessage (rejected/plain string)."""
    return tool.invoke({"name": tool.name, "args": args, "id": "call_test", "type": "tool_call"})


def _update(res) -> dict:
    """The state update a tool's Command carries, or {} for a plain ToolMessage."""
    return getattr(res, "update", {}) or {}


def _content(res) -> str:
    return getattr(res, "content", None) or str(res)


def test_submit_repro_confirmed_when_script_fails():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="ValueError")
    tool = make_submit_repro_tool(sandbox=sandbox)
    res = _call(tool, script="raise ValueError()", error_output="ValueError")
    repro = _update(res).get("repro")
    assert repro and repro["confirmed"] is True and repro["repro_script"] == "raise ValueError()"


def test_submit_repro_rejected_when_script_passes():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="", stderr="")
    tool = make_submit_repro_tool(sandbox=sandbox)
    res = _call(tool, script="print('ok')", error_output="")
    assert "not reproduced" in _content(res).lower()
    assert not _update(res)  # nothing written to the channel


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
    sandbox = _sandbox_with_file("def add(a, b):\n    return a - b  # bug\n")
    tool = make_apply_patch_tool(sandbox=sandbox)
    res = _call(tool, path="calc.py", old_content="return a - b  # bug", new_content="return a + b")
    patches = _update(res).get("patches")
    assert patches and patches[0]["path"] == "calc.py"
    sandbox.write_file.assert_called_once()


def test_apply_patch_old_content_not_found():
    sandbox = _sandbox_with_file("def add(a, b):\n    return a - b  # bug\n")
    tool = make_apply_patch_tool(sandbox=sandbox)
    res = _call(tool, path="calc.py", old_content="this does not exist in the file", new_content="x")
    assert "error" in _content(res).lower()
    assert not _update(res)


def test_apply_patch_file_not_found():
    sandbox = MagicMock()
    sandbox.exec.return_value = RunResult(exit_code=1, stdout="", stderr="")  # test -f fails
    tool = make_apply_patch_tool(sandbox=sandbox)
    res = _call(tool, path="ghost.py", old_content="x", new_content="y")
    assert "not found" in _content(res).lower()


def test_submit_fix_uses_repro_as_oracle_and_captures_git_diff():
    """No benchmark test in production: submit_fix accepts when the reproduction
    passes, and records the real `git diff` as the patch."""
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="ok", stderr="")  # repro passes
    sandbox.git.return_value = RunResult(exit_code=0, stdout="THE REAL DIFF", stderr="")
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="fixed it")
    fix = _update(res).get("fix")
    # Patch is normalized to end with a newline (git apply requires it).
    assert fix and fix["patch"] == "THE REAL DIFF\n"
    # The captured patch comes from `git diff` (other git calls may follow for the
    # regression gate's name-only listing / stash).
    sandbox.git.assert_any_call("diff")


def test_submit_fix_rejected_when_diff_is_empty():
    """A passing repro with no source change isn't a fix — reject the empty diff."""
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="ok", stderr="")
    sandbox.git.return_value = RunResult(exit_code=0, stdout="  \n", stderr="")  # empty diff
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="x")
    assert "no source changes" in _content(res).lower()
    assert not _update(res)


def test_submit_fix_rejected_when_repro_still_fails():
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="boom")  # repro fails
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="x")
    assert "cannot submit" in _content(res).lower()
    assert not _update(res)
    sandbox.git.assert_not_called()


def _gate_sandbox(baseline_out, post_out,
                  changed="pkg/foo.py\n",
                  test_file="/workspace/repo/tests/test_foo.py"):
    """A sandbox mock that drives the submit_fix regression gate: repro passes,
    one changed source module maps to one existing test file, and the two pytest
    runs (baseline before stash-pop, post after) return the given summaries."""
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="ok", stderr="")
    calls = {"pytest": 0}

    def exec_side(cmd, workdir=None):
        if "pytest" in cmd:
            calls["pytest"] += 1
            out = baseline_out if calls["pytest"] == 1 else post_out
            return RunResult(exit_code=0, stdout=out, stderr="")
        if cmd.startswith("find "):
            return RunResult(exit_code=0, stdout=test_file + "\n", stderr="")
        return RunResult(exit_code=0, stdout="", stderr="")
    sandbox.exec.side_effect = exec_side

    def git_side(*args, workdir=None):
        if args[:1] == ("diff",) and "--name-only" in args:
            return RunResult(exit_code=0, stdout=changed, stderr="")
        if args[:1] == ("diff",):
            return RunResult(exit_code=0, stdout="diff --git a/pkg/foo.py b/pkg/foo.py\n+x", stderr="")
        if args[:1] == ("stash",):
            return RunResult(exit_code=0, stdout="Saved working directory", stderr="")
        return RunResult(exit_code=0, stdout="", stderr="")
    sandbox.git.side_effect = git_side
    return sandbox


def test_submit_fix_regression_gate_blocks_newly_failing_test():
    """A patch that passes the repro but makes a previously-passing test fail is
    rejected with the regressing test names."""
    sandbox = _gate_sandbox(
        baseline_out="",  # buggy code: the module's tests all pass
        post_out="FAILED tests/test_foo.py::test_neighbour - AssertionError",
    )
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="x")
    assert "regression" in _content(res).lower()
    assert "test_neighbour" in _content(res)
    assert not _update(res)  # fix not accepted


def test_submit_fix_regression_gate_baselines_preexisting_failures():
    """A test already failing at the buggy commit is baselined out — it must not
    block the fix when it's still failing afterwards."""
    line = "FAILED tests/test_foo.py::test_flaky - X"
    sandbox = _gate_sandbox(baseline_out=line, post_out=line)
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="x")
    fix = _update(res).get("fix")
    assert fix and fix["patch"].startswith("diff --git")


def test_submit_fix_regression_gate_passes_when_clean():
    sandbox = _gate_sandbox(baseline_out="", post_out="")
    tool = make_submit_fix_tool(sandbox=sandbox, repro_script="s")
    res = _call(tool, summary="x")
    assert _update(res).get("fix")
    assert "regression check passed" in _content(res).lower()


def test_submit_root_cause_parses_newline_alternatives():
    """alternatives can be a newline string (bullets stripped) -> ranked list."""
    tool = make_submit_root_cause_tool()
    res = _call(
        tool, summary="s", relevant_lines="foo.py:1", hypothesis="primary",
        evidence="observed", fix_plan="do X",
        alternatives="- first alt\n- second alt",
    )
    rc = _update(res).get("root_cause")
    assert rc["alternatives"] == ["first alt", "second alt"]


def test_submit_root_cause_alternatives_default_empty():
    tool = make_submit_root_cause_tool()
    res = _call(
        tool, summary="s", relevant_lines=["foo.py:1"], hypothesis="h",
        evidence="e", fix_plan="p",
    )
    assert _update(res).get("root_cause")["alternatives"] == []


def test_shell_runs_command_and_returns_exit_code_and_output():
    sandbox = MagicMock()
    sandbox.exec.return_value = RunResult(exit_code=0, stdout="hello\n", stderr="")
    tool = make_shell_tool(sandbox=sandbox)
    out = tool.invoke({"command": "echo hello"})
    sandbox.exec.assert_called_once_with("echo hello")
    assert "exit_code=0" in out and "hello" in out


def test_apply_patch_refuses_test_files():
    """The fix agent must not edit the test that serves as its success oracle."""
    sandbox = MagicMock()
    tool = make_apply_patch_tool(sandbox=sandbox)
    for path in ("test/units/test_x.py", "tests/test_foo.py", "pkg/foo/bar_test.py"):
        res = _call(tool, path=path, old_content="a", new_content="b")
        assert "refused" in _content(res).lower() and "test" in _content(res).lower()
        assert not _update(res)
    sandbox.exec.assert_not_called()       # rejected before any filesystem access
    sandbox.write_file.assert_not_called()


# ---------------------------------------------------------------------------
# submit_culprit (bisect) — must reject empty / unresolvable SHAs so a bogus
# culprit never propagates to the manager (which would advance on it).
# ---------------------------------------------------------------------------

from unittest.mock import patch
from autodebug.tools.bisect import make_submit_culprit_tool
from autodebug.tools.git_utils import CommitInfo


def _commit_info(sha: str) -> CommitInfo:
    return CommitInfo(sha=sha, short_sha=sha[:7], message="msg" if sha else "",
                      author="a", date="d", diff="diff" if sha else "")


def test_submit_culprit_records_a_resolvable_sha():
    tool = make_submit_culprit_tool(sandbox=MagicMock())
    with patch("autodebug.tools.git_utils.get_commit_info",
               return_value=_commit_info("abc123def456")):
        res = _call(tool, sha="abc123", explanation="introduced the bug")
    bisect = _update(res).get("bisect")
    assert bisect and bisect["culprit_commit"] == "abc123def456"


def test_submit_culprit_rejects_blank_sha():
    tool = make_submit_culprit_tool(sandbox=MagicMock())
    res = _call(tool, sha="   ", explanation="dunno")
    assert "no sha" in _content(res).lower()
    assert not _update(res)  # nothing written -> agent stays in its loop and retries cheaply


def test_submit_culprit_rejects_unresolvable_sha():
    tool = make_submit_culprit_tool(sandbox=MagicMock())
    with patch("autodebug.tools.git_utils.get_commit_info",
               return_value=_commit_info("")):  # git couldn't resolve it
        res = _call(tool, sha="deadbeef", explanation="guess")
    assert "does not resolve" in _content(res).lower()
    assert not _update(res)


# ---------------------------------------------------------------------------
# git_utils.restore_checkout — aborts any in-progress bisect and returns to HEAD
# ---------------------------------------------------------------------------

def test_restore_checkout_resets_bisect_then_checks_out():
    from autodebug.tools import git_utils
    sandbox = MagicMock()
    git_utils.restore_checkout(sandbox, "abc123")
    calls = [c.args for c in sandbox.git.call_args_list]
    assert ("bisect", "reset") in calls
    assert ("checkout", "--force", "abc123") in calls


def test_restore_checkout_swallows_errors():
    from autodebug.tools import git_utils
    sandbox = MagicMock()
    sandbox.git.side_effect = RuntimeError("git blew up")
    git_utils.restore_checkout(sandbox, "abc123")  # must not raise (used in a finally)


def test_reset_worktree_hard_resets_to_head():
    from autodebug.tools import git_utils
    sandbox = MagicMock()
    git_utils.reset_worktree(sandbox)
    assert ("reset", "--hard", "HEAD") in [c.args for c in sandbox.git.call_args_list]


def test_reset_worktree_swallows_errors():
    from autodebug.tools import git_utils
    sandbox = MagicMock()
    sandbox.git.side_effect = RuntimeError("git blew up")
    git_utils.reset_worktree(sandbox)  # must not raise
