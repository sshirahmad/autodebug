"""Tools for the Fix agent."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from autodebug.sandbox import Sandbox
from autodebug.state import FixResult


def make_apply_patch_tool(repo_path: Path, patches: list, **_):
    @tool
    def apply_patch(path: str, old_content: str, new_content: str) -> str:
        """Apply a code change by replacing old content with new content in a file."""
        fp = repo_path / path
        if not fp.exists():
            return f"File not found: {path}"
        original = fp.read_text(encoding="utf-8")
        if old_content not in original:
            return "ERROR: old_content not found verbatim in file. Read the file first to get exact content."
        fp.write_text(original.replace(old_content, new_content, 1), encoding="utf-8")
        patches.append({"path": path, "old": old_content, "new": new_content})
        return f"Patch applied to {path}"
    return apply_patch


def make_run_repro_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool
    def run_repro() -> str:
        """Run the reproduction script — should PASS after fix is applied."""
        run = sandbox.run_script(repro_script)
        return f"exit_code={run.exit_code}\n{run.output[-3000:]}"
    return run_repro


def make_run_tests_tool(sandbox: Sandbox, test_command: str = "", **_):
    @tool
    def run_tests(test_path: str = "") -> str:
        """Run the targeted test suite to check for regressions.

        Args:
            test_path: Optional path to a test file or directory. If omitted,
                       uses the benchmark test command if one was provided.
        """
        cmd = test_command if (not test_path and test_command) else f"python -m pytest {test_path} -x --tb=short -q"
        run = sandbox.run(cmd)
        return f"exit_code={run.exit_code}\n{run.output[-4000:]}"
    return run_tests


def make_submit_fix_tool(sandbox: Sandbox, repro_script: str, test_command: str, patches: list, result: list, **_):
    @tool
    def submit_fix(summary: str) -> str:
        """Submit the final validated fix once the repro passes and targeted tests pass."""
        repro_run = sandbox.run_script(repro_script)
        if not repro_run.success:
            return f"Cannot submit: repro still fails.\n{repro_run.output[-2000:]}"
        test_output = ""
        if test_command:
            test_run = sandbox.run(test_command)
            test_output = test_run.output[-2000:]
            if not test_run.success:
                return f"Cannot submit: targeted tests failing.\n{test_output}"
        result.append(FixResult(
            patch=_build_patch_summary(patches),
            attempts=len(patches),
            test_output=test_output,
        ))
        return "Fix validated and submitted."
    return submit_fix


def _build_patch_summary(patches: list[dict]) -> str:
    lines = []
    for p in patches:
        lines.append(f"--- {p['path']}")
        lines.append(f"+++ {p['path']}")
        for line in p["old"].splitlines():
            lines.append(f"-{line}")
        for line in p["new"].splitlines():
            lines.append(f"+{line}")
        lines.append("")
    return "\n".join(lines)
