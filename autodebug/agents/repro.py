"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent, BudgetExceeded
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert software debugger. Your job is to reproduce a bug
given a bug report and access to the repository.

You will be given tools to:
- read files from the repository
- run Python code in a sandbox
- write a reproduction script

Your goal: write the MINIMAL Python script that triggers the reported bug.
The script must fail with a clear error when run against the current HEAD.

Think step by step:
1. Read the bug report carefully
2. Explore relevant source files to understand the code
3. Write a minimal script that triggers the bug
4. Run it to confirm it fails
5. If it doesn't fail, revise and retry

When you have a confirmed reproducing script, call the `submit_repro` tool.\
"""

class ReproAgent(BaseAgent):
    def __init__(self, *, tool_builder: Callable, system_prompt=None, **kwargs):
        super().__init__(**kwargs)
        self._tool_builder = tool_builder
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))
        result: list = []

        tools = self._tool_builder(repo_path=repo_path, sandbox=sandbox, result=result)
        tool_map = {t.name: t for t in tools}

        for retry in range(self._max_retries + 1):
            budget = self._new_budget()
            result.clear()
            messages: list[BaseMessage] = [
                self.human(
                    f"Bug report:\n\n{state.bug_report}\n\n"
                    "Please reproduce this bug. Start by exploring the repository structure."
                )
            ]
            try:
                while True:
                    response = self._chat(messages, tools=tools, system=self._system_prompt)
                    state.total_llm_calls += 1
                    inp, out = self._usage(response)
                    state.total_tokens += inp + out
                    state.total_cost += budget.add_tokens(inp, out)
                    budget.check()

                    if not response.tool_calls:
                        break

                    tool_results: list[BaseMessage] = []
                    for tc in response.tool_calls:
                        tool_results.append(self.invoke_tool(tool_map, tc))

                    messages.append(response)
                    messages.extend(tool_results)

                    if result:
                        state.repro = result[0]
                        state.stage = PipelineStage.BISECT
                        return state

            except BudgetExceeded:
                if retry < self._max_retries:
                    continue
                state.stage = PipelineStage.FAILED
                state.error = f"ReproAgent: budget exceeded after {retry + 1} attempt(s)"
                return state

        state.stage = PipelineStage.FAILED
        state.error = "ReproAgent failed to reproduce the bug"
        return state
