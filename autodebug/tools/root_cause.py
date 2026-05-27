"""Tools for the Root Cause agent."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from autodebug.sandbox import Sandbox
from autodebug.state import RootCauseResult
from autodebug.tools import git_utils


def make_read_file_at_parent_tool(repo_path: Path, parent_sha: str, **_):
    @tool
    def read_file_at_parent(path: str) -> str:
        """Read a file as it was BEFORE the culprit commit (to compare)."""
        content = git_utils.get_file_at_commit(str(repo_path), parent_sha, path)
        return content or f"File not found at parent commit: {path}"
    return read_file_at_parent


def make_run_repro_with_traceback_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool
    def run_repro_with_traceback() -> str:
        """Run the reproduction script and capture the full Python traceback."""
        wrapper = (
            "import traceback, sys\n"
            "try:\n"
            "    exec(open('/workspace/repro.py').read())\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
        )
        run = sandbox.run(
            command="python /workspace/script.py",
            extra_files={"script.py": wrapper, "repro.py": repro_script},
        )
        return run.output[-4000:]
    return run_repro_with_traceback


def make_submit_root_cause_tool(result: list, **_):
    @tool
    def submit_root_cause(summary: str, relevant_lines: str | list[str], hypothesis: str) -> str:
        """Submit the root cause analysis.

        Args:
            summary: One-paragraph description of the root cause.
            relevant_lines: List of relevant file:line references, e.g.
                ["lib/foo/bar.py:42", "lib/foo/bar.py:55"]. May also be
                passed as a newline-separated string.
            hypothesis: Explanation of why the change caused the bug.
        """
        if isinstance(relevant_lines, str):
            relevant_lines = [l.strip() for l in relevant_lines.splitlines() if l.strip()]
        result.append(RootCauseResult(
            summary=summary,
            relevant_lines=relevant_lines,
            hypothesis=hypothesis,
        ))
        return "Root cause submitted."
    return submit_root_cause
