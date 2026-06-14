"""Tools for the Bisect agent: mark_bad, mark_good, mark_skip, submit_result."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from autodebug.sandbox import Sandbox
from autodebug.state import BisectResult


def make_mark_bad_tool(verdict: list, **_):
    @tool
    def mark_bad() -> str:
        """Mark the current commit as BAD (bug is present)."""
        verdict.append("bad")
        return "Marked as bad."
    return mark_bad


def make_mark_good_tool(verdict: list, **_):
    @tool
    def mark_good() -> str:
        """Mark the current commit as GOOD (bug is NOT present)."""
        verdict.append("good")
        return "Marked as good."
    return mark_good


def make_mark_skip_tool(verdict: list, **_):
    @tool
    def mark_skip() -> str:
        """Skip this commit (ambiguous or environment error)."""
        verdict.append("skip")
        return "Marked as skip."
    return mark_skip


def make_submit_result_tool(sha: str, info, result: list, **_):
    @tool(parse_docstring=True)
    def submit_result(culprit_sha: str, explanation: str) -> str:
        """Submit the final culprit commit when bisect is complete.

        Args:
            culprit_sha: SHA of the commit that introduced the bug.
            explanation: Brief explanation of why this commit is the culprit.
        """
        result.append(BisectResult(
            culprit_commit=sha,
            commit_message=info.message,
            commit_diff=info.diff,
            steps_taken=0,
        ))
        return "Bisect result submitted."
    return submit_result


def make_submit_culprit_tool(sandbox: Sandbox, **_):
    from autodebug.tools import git_utils as _gu

    @tool(parse_docstring=True)
    def submit_culprit(
        sha: str, explanation: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command | str:
        """Submit the SHA of the commit that introduced the bug.

        Args:
            sha: The full or short commit SHA of the culprit commit.
            explanation: Brief explanation of why this commit is the culprit.
        """
        # Validate BEFORE recording: an empty or unresolvable SHA must NOT be
        # accepted as a culprit (returning a string leaves the channel unset, so
        # the agent keeps iterating instead of advancing on a bogus culprit).
        sha = (sha or "").strip()
        if not sha:
            return ("No SHA provided. Identify the culprit first (e.g. "
                    "`git log --oneline <known_good>..HEAD`), then call submit_culprit "
                    "with its SHA.")
        info = _gu.get_commit_info(sandbox, sha)
        if not info.sha:
            return (f"'{sha}' does not resolve to a commit in this repo. Verify it "
                    "with `git show <sha>` and submit a real culprit SHA.")
        bisect = BisectResult(
            culprit_commit=info.sha,
            commit_message=info.message,
            commit_diff=info.diff,
            steps_taken=0,
        )
        return Command(update={
            "bisect": bisect.model_dump(),
            "messages": [ToolMessage(f"Culprit commit recorded: {info.sha} — {info.message}",
                                     tool_call_id=tool_call_id)],
        })
    return submit_culprit
