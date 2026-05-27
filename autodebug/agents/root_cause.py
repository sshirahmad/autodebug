"""Root Cause Agent — explains WHY the culprit commit broke things."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent, BudgetExceeded
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineer performing a root cause analysis.

You have:
1. A commit diff that introduced a regression
2. The output of running the reproduction script
3. Access to the repository files for deeper investigation

Your job: produce a precise root cause analysis that explains:
- What changed in the culprit commit
- Why that change caused the bug
- Which specific lines are responsible

Use the tools to read any files you need for context.
When you have enough information, call `submit_root_cause`.\
"""

class RootCauseAgent(BaseAgent):
    def __init__(self, *, tool_builder: Callable, system_prompt=None, **kwargs):
        super().__init__(**kwargs)
        self._tool_builder = tool_builder
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.bisect
        assert state.repro

        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))
        result: list = []

        tools = self._tool_builder(
            repo_path=repo_path,
            sandbox=sandbox,
            parent_sha=f"{state.bisect.culprit_commit}^",
            repro_script=state.repro.repro_script,
            result=result,
        )
        tool_map = {t.name: t for t in tools}

        initial_run = sandbox.run_script(state.repro.repro_script)
        initial_context = (
            f"Culprit commit: {state.bisect.commit_message}\n"
            f"SHA: {state.bisect.culprit_commit}\n\n"
            f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
            f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n\n"
            "Please identify the root cause of this regression."
        )

        for retry in range(self._max_retries + 1):
            budget = self._new_budget()
            result.clear()
            messages: list[BaseMessage] = [self.human(initial_context)]
            try:
                while True:
                    response = self._chat(messages, tools=tools, system=self._system_prompt)
                    state.total_llm_calls += 1
                    tokens = self.count_tokens(response)
                    state.total_tokens += tokens
                    budget.add_tokens(tokens)
                    budget.check()

                    if not response.tool_calls:
                        break

                    tool_results: list[BaseMessage] = []
                    for tc in response.tool_calls:
                        tool_results.append(self.invoke_tool(tool_map, tc))

                    messages.append(response)
                    messages.extend(tool_results)

                    if result:
                        state.root_cause = result[0]
                        state.stage = PipelineStage.FIX
                        return state

            except BudgetExceeded:
                if retry < self._max_retries:
                    continue
                state.stage = PipelineStage.FAILED
                state.error = f"RootCauseAgent: budget exceeded after {retry + 1} attempt(s)"
                return state

        state.stage = PipelineStage.FAILED
        state.error = "RootCauseAgent: could not determine root cause"
        return state
