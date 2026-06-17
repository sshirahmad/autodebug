"""Tools for the Root Cause agent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

import re

from autodebug.sandbox import Sandbox
from autodebug.state import RootCauseResult
from autodebug.tools import git_utils
from autodebug.tools.introspect import inspect_harness, postmortem_harness


def make_read_file_at_parent_tool(sandbox: Sandbox, parent_sha: str, **_):
    @tool(parse_docstring=True)
    def read_file_at_parent(path: str) -> str:
        """Read a file as it was BEFORE the culprit commit (to compare).

        Args:
            path: Relative path from the repo root to read at the parent commit.
        """
        if not parent_sha:
            return ("No culprit commit was identified (bisect was inconclusive), so "
                    "there is no parent revision to compare against. Use read_file "
                    "and the runtime-inspection tools instead.")
        content = git_utils.get_file_at_commit(sandbox, parent_sha, path)
        return content or f"File not found at parent commit: {path}"
    return read_file_at_parent


def make_run_repro_with_traceback_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool(parse_docstring=True)
    def run_repro_with_traceback(script: str = "") -> str:
        """Run a script (default: the reproduction) and, on any uncaught exception,
        return the full traceback PLUS the local variables at every stack frame —
        so you see the real failure state (values, types), not just the message.

        Args:
            script: Python to run. Omit to run the reproduction script.
        """
        run = sandbox.run_script(postmortem_harness(script or repro_script))
        out = run.output[:4000]  # keep the head: exception + innermost frames
        return out or "Script completed with no uncaught exception (exit 0)."
    return run_repro_with_traceback


def make_inspect_at_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool(parse_docstring=True)
    def inspect_at(location: str, expressions: str, driver: str = "") -> str:
        """Probe program state at a specific source line — set a tracepoint, run,
        and see the actual values there instead of guessing.

        Runs `driver` (default: the reproduction) and, each time `location` is
        reached, evaluates your expressions in that frame and reports them along
        with the frame's local variables (first few hits).

        Args:
            location: Where to probe — a repo-relative file path and a line
                number separated by a colon, for example
                lib/ansible/galaxy/collection.py then 670.
            expressions: Expressions to evaluate at that line, comma- or
                newline-separated (for example worker_count, then a dev-shm check).
            driver: Python that triggers the code path (sets up inputs and calls
                it). Omit to use the reproduction script.
        """
        try:
            file_part, line_part = location.rsplit(":", 1)
            target_line = int(line_part.strip())
        except Exception:
            return "location must be 'relative/path/to/file.py:LINE'."
        exprs = [e.strip() for e in re.split(r"[,\n]", expressions) if e.strip()]
        harness = inspect_harness(file_part.strip(), target_line, exprs, driver or repro_script)
        run = sandbox.run_script(harness)
        out = run.output.strip()
        if out in ("", "[]"):
            return (
                f"No hits — {file_part}:{target_line} was never reached. Check the "
                "path/line and that the driver actually triggers that code path."
            )
        return out[-4000:]
    return inspect_at


def make_submit_root_cause_tool(**_):
    @tool(parse_docstring=True)
    def submit_root_cause(
        summary: str, relevant_lines: str | list[str], hypothesis: str,
        evidence: str, fix_plan: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Submit the root cause analysis once you have OBSERVED the failure.

        Args:
            summary: One-paragraph description of the root cause.
            relevant_lines: The responsible code locations as path-then-line
                references (for example lib/foo/bar.py line 42). Pass a list of
                such strings, or a single newline-separated string.
            hypothesis: Explanation of why the change caused the bug.
            evidence: The runtime evidence you OBSERVED via run_repro_with_traceback
                or inspect_at — the exception, the failing line, the values you saw.
                Quote the tool output; do not fabricate.
            fix_plan: The CONCRETE change the fixer must make to resolve the bug —
                which file and function, the exact lines/block to change, and the
                precise new logic (e.g. wrap which call in try/except, what the
                fallback does). Specific enough to implement WITHOUT re-investigating.
        """
        if isinstance(relevant_lines, str):
            relevant_lines = [l.strip() for l in relevant_lines.splitlines() if l.strip()]
        rc = RootCauseResult(
            summary=summary,
            relevant_lines=relevant_lines,
            hypothesis=hypothesis,
            evidence=evidence,
            fix_plan=fix_plan,
        )
        return Command(update={
            "root_cause": rc.model_dump(),
            "messages": [ToolMessage("Root cause submitted.", tool_call_id=tool_call_id)],
        })
    return submit_root_cause
