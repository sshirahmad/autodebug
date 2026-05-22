"""Bisect Agent — finds the commit that introduced a regression using git bisect."""

from __future__ import annotations

import os

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent
from autodebug.sandbox import Sandbox
from autodebug.state import BisectResult, DebugState, PipelineStage
from autodebug.tools import git_utils

MAX_STEPS = int(os.getenv("BISECT_MAX_STEPS", "30"))
FALLBACK_LOOKBACK_DAYS = [7, 30, 90, 180]

SYSTEM_PROMPT = """You are an expert at debugging software regressions.

You are given a reproduction script that FAILS on the current HEAD (bug is present).
Your job is to classify whether the bug is present in the current commit.

Rules:
- If the repro FAILS (non-zero exit or expected error): bug IS present → call `mark_bad`
- If the repro PASSES (exit 0, no error): bug is NOT present → call `mark_good`
- If the result is ambiguous (flaky test, environment error, unrelated failure): call `mark_skip`
- When bisect is complete (git says "is the first bad commit"): call `submit_result`

Be conservative: only skip if confident the result is unrelated to the bug."""

TOOLS = [
    {
        "name": "mark_bad",
        "description": "Mark the current commit as BAD (bug is present)",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "mark_good",
        "description": "Mark the current commit as GOOD (bug is NOT present)",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "mark_skip",
        "description": "Skip this commit (ambiguous or environment error)",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_result",
        "description": "Submit the final culprit commit when bisect is complete",
        "parameters": {
            "type": "object",
            "properties": {
                "culprit_sha": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["culprit_sha", "explanation"],
        },
    },
]


class BisectAgent(BaseAgent):
    """Runs git bisect guided by LLM classification of repro results."""

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.repro and state.repro.confirmed

        repo = state.repo_local_path
        repro_script = state.repro.repro_script
        sandbox = Sandbox(repo)

        git_utils.unshallow(repo)

        good_ref = state.known_good_commit or self._find_good_commit(repo, repro_script, sandbox)
        if not good_ref:
            state.stage = PipelineStage.FAILED
            state.error = "BisectAgent: could not find a known-good commit to start bisect"
            return state

        n_commits = git_utils.count_commits_between(repo, good_ref, "HEAD")

        git_utils.bisect_reset(repo)
        git_utils.bisect_start(repo)
        git_utils.bisect_bad(repo, "HEAD")
        bisect_output = git_utils.bisect_good(repo, good_ref)

        culprit: BisectResult | None = None

        for step in range(MAX_STEPS):
            sha = git_utils.current_sha(repo)
            info = git_utils.get_commit_info(repo, sha)

            if "is the first bad commit" in bisect_output:
                culprit = BisectResult(
                    culprit_commit=sha,
                    commit_message=info.message,
                    commit_diff=info.diff,
                    steps_taken=step,
                )
                break

            run_result = sandbox.run_script(repro_script)
            verdict, bisect_output, submitted = self._classify(
                state, sha, info, run_result.output, step, n_commits
            )

            if verdict == "submit" and submitted:
                culprit = submitted
                break
            elif verdict == "bad":
                bisect_output = git_utils.bisect_bad(repo)
            elif verdict == "good":
                bisect_output = git_utils.bisect_good(repo)
            else:
                bisect_output = git_utils.bisect_skip(repo)

            if "is the first bad commit" in bisect_output:
                sha = git_utils.current_sha(repo)
                info = git_utils.get_commit_info(repo, sha)
                culprit = BisectResult(
                    culprit_commit=sha,
                    commit_message=info.message,
                    commit_diff=info.diff,
                    steps_taken=step + 1,
                )
                break

        git_utils.bisect_reset(repo)

        if culprit:
            state.bisect = culprit
            state.stage = PipelineStage.ROOT_CAUSE
        else:
            state.stage = PipelineStage.FAILED
            state.error = f"BisectAgent: did not converge after {MAX_STEPS} steps"

        return state

    def _find_good_commit(
        self, repo: str, repro_script: str, sandbox: Sandbox
    ) -> str | None:
        for days in FALLBACK_LOOKBACK_DAYS:
            ref = git_utils.find_commit_before_days(repo, days)
            if not ref:
                continue
            git_utils.checkout(repo, ref)
            result = sandbox.run_script(repro_script)
            git_utils.checkout(repo, "HEAD")
            if result.success:
                return ref
        return None

    def _classify(
        self, state: DebugState, sha: str, info, output: str, step: int, total: int
    ) -> tuple[str, str, BisectResult | None]:
        messages: list[BaseMessage] = [
            self.human(
                f"Bisect step {step + 1} (~{total} commits in range)\n\n"
                f"Commit: {info.short_sha} — {info.message}\n"
                f"Author: {info.author}  Date: {info.date}\n\n"
                f"Repro output:\n```\n{output[-3000:]}\n```\n\n"
                "Is the bug present in this commit?"
            )
        ]

        response = self._chat(messages, tools=TOOLS, system=SYSTEM_PROMPT)
        state.total_llm_calls += 1
        state.total_tokens += self.count_tokens(response)

        for tc in response.tool_calls:
            if tc["name"] == "mark_bad":
                return "bad", "", None
            if tc["name"] == "mark_good":
                return "good", "", None
            if tc["name"] == "mark_skip":
                return "skip", "", None
            if tc["name"] == "submit_result":
                return "submit", "", BisectResult(
                    culprit_commit=sha,
                    commit_message=info.message,
                    commit_diff=info.diff,
                    steps_taken=step + 1,
                )

        return "bad", "", None
