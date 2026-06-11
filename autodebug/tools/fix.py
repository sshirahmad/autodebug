"""Tools for the Fix agent."""

from __future__ import annotations

import shlex

from langchain_core.tools import tool

from autodebug.patch_utils import is_test_path
from autodebug.sandbox import REPO_DIR, Sandbox
from autodebug.state import FixResult


def make_apply_patch_tool(sandbox: Sandbox, patches: list, **_):
    @tool(parse_docstring=True)
    def apply_patch(path: str, old_content: str, new_content: str) -> str:
        """Apply a code change by replacing old content with new content in a file.

        Args:
            path: Relative path from the repo root to the file to edit (not a test file).
            old_content: Exact text to replace — must match the file verbatim,
                whitespace included.
            new_content: Replacement text.
        """
        # Never let the fix touch test files: the targeted test is the success
        # oracle (submit_fix runs it), so editing it would let the agent "pass"
        # by changing the test instead of fixing the source.
        if is_test_path(path):
            return (
                f"Refused: '{path}' is a test file. Patch the source code, not the "
                "test — the test is the success criterion and must stay unchanged."
            )
        full = f"{REPO_DIR}/{path.lstrip('/')}"
        check = sandbox.exec(f"test -f {shlex.quote(full)}")
        if check.exit_code != 0:
            return f"File not found: {path}"
        original = sandbox.exec(f"cat {shlex.quote(full)}").stdout
        if old_content not in original:
            return (
                "ERROR: old_content not found verbatim in file. "
                "Read the file first to get exact content."
            )
        updated = original.replace(old_content, new_content, 1)
        sandbox.write_file(path, updated)
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
    @tool(parse_docstring=True)
    def run_tests(test_path: str = "") -> str:
        """Run the project's own tests with pytest to check for regressions.

        Args:
            test_path: Path to a test file or directory to run (relative to the
                repo root). Omit to run the whole suite (may be slow).
        """
        cmd = f"python -m pytest {test_path} -x --tb=short -q"
        run = sandbox.exec(cmd)
        return f"exit_code={run.exit_code}\n{run.output[-4000:]}"
    return run_tests


def make_submit_fix_tool(
    sandbox: Sandbox, repro_script: str, patches: list, result: list, **_,
):
    @tool(parse_docstring=True)
    def submit_fix(summary: str) -> str:
        """Submit the fix once the reproduction passes.

        The reproduction is the success criterion: there is no benchmark test in
        production, so a fix is accepted when the repro script exits 0.

        Args:
            summary: One-line description of the fix you applied.
        """
        repro_run = sandbox.run_script(repro_script)
        if not repro_run.success:
            return f"Cannot submit: the reproduction still fails.\n{repro_run.output[-2000:]}"
        # Capture the actual source changes as a real, applicable diff so the fix
        # can be validated independently later — not just a textual summary.
        diff = sandbox.git("diff").stdout
        if not diff.strip():
            return (
                "Cannot submit: there are no source changes (empty diff). The "
                "reproduction passing without a code change means it isn't testing "
                "the bug — revise the fix (or the reproduction) before submitting."
            )
        # git apply requires a trailing newline; normalize so the patch is valid.
        diff = diff.rstrip("\n") + "\n"
        result.append(FixResult(
            patch=diff,
            attempts=len(patches),
            test_output="",
        ))
        return "Fix validated against the reproduction and submitted."
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
