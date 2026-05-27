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


def make_run_tests_tool(sandbox: Sandbox, **_):
    @tool
    def run_tests(test_path: str = "") -> str:
        """Run the test suite to check for regressions."""
        run = sandbox.run(f"pip install -e . -q && python -m pytest {test_path} -x --tb=short -q")
        return f"exit_code={run.exit_code}\n{run.output[-4000:]}"
    return run_tests


def make_submit_fix_tool(sandbox: Sandbox, repro_script: str, patches: list, result: list, **_):
    @tool
    def submit_fix(summary: str) -> str:
        """Submit the final validated fix once repro passes and tests pass."""
        repro_run = sandbox.run_script(repro_script)
        test_run = sandbox.run("python -m pytest --tb=short -q")
        if not repro_run.success:
            return f"Cannot submit: repro still fails.\n{repro_run.output[-2000:]}"
        if not test_run.success:
            return f"Cannot submit: test suite failing.\n{test_run.output[-2000:]}"
        result.append(FixResult(
            patch=_build_patch_summary(patches),
            attempts=len(patches),
            test_output=test_run.output[-2000:],
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
