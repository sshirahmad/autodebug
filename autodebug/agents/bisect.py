"""Bisect Agent — finds the commit that introduced a regression."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent, BudgetExceeded
from autodebug.sandbox import Sandbox
from autodebug.state import BisectResult, DebugState, PipelineStage
from autodebug.tools import git_utils
from autodebug.tools.bisect import make_submit_culprit_tool


class BisectAgent(BaseAgent):
    def __init__(self, *, tool_builder: Callable, system_prompt=None, **kwargs):
        super().__init__(**kwargs)
        self._tool_builder = tool_builder
        self._system_prompt = system_prompt

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.repro and state.repro.confirmed

        repo = str(state.repo_local_path)
        sandbox = Sandbox(repo)

        git_utils.unshallow(repo)
        original_sha = git_utils.current_sha(repo)
        head_info = git_utils.get_commit_info(repo, original_sha)

        result: list[BisectResult] = []

        # Build tools: run_script + read_file (from config) + submit_culprit
        tools = self._tool_builder(
            repo_path=Path(repo),
            sandbox=sandbox,
            repo=repo,
            result=result,
        )
        tool_map = {t.name: t for t in tools}

        initial_msg = self.human(
            f"Bug report:\n{state.bug_report}\n\n"
            f"Current HEAD: {head_info.short_sha} — {head_info.message}\n"
            f"Date: {head_info.date}\n\n"
            f"Repro script (fails at HEAD, exits non-zero when bug is present):\n"
            f"```python\n{state.repro.repro_script}\n```\n\n"
            f"Repro error output:\n```\n{state.repro.error_output[:2000]}\n```\n\n"
            "Use git log/blame/show (via run_script with subprocess) to find the commit "
            "that introduced this bug, then call `submit_culprit` with its SHA."
        )

        for retry in range(self._max_retries + 1):
            budget = self._new_budget()
            result.clear()
            messages: list[BaseMessage] = [initial_msg]
            try:
                while True:
                    response = self._chat(messages, tools=tools, system=self._system_prompt)
                    tokens = self.count_tokens(response)
                    state.total_llm_calls += 1
                    state.total_tokens += tokens
                    budget.add_tokens(tokens)
                    budget.check()

                    if not response.tool_calls:
                        break

                    tool_results = [self.invoke_tool(tool_map, tc) for tc in response.tool_calls]
                    messages.append(response)
                    messages.extend(tool_results)

                    if result:
                        state.bisect = result[0]
                        state.stage = PipelineStage.ROOT_CAUSE
                        return state

            except BudgetExceeded:
                if retry < self._max_retries:
                    continue
                state.stage = PipelineStage.FAILED
                state.error = "BisectAgent: budget exceeded"
                return state

        state.stage = PipelineStage.FAILED
        state.error = "BisectAgent: could not identify culprit commit"
        return state
