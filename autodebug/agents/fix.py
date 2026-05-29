"""Fix Agent — generates and validates patches using an InspectCoder-style loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent, BudgetExceeded
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineer fixing a confirmed regression bug.

You have:
1. The root cause analysis explaining WHY the bug was introduced
2. The culprit commit diff
3. The reproduction script that currently FAILS
4. Access to the full repository

Your job: write a patch that fixes the bug such that:
- The reproduction script PASSES
- The existing test suite still PASSES (no regressions)

Workflow:
1. Read relevant files to understand the code fully
2. Write a targeted patch with `apply_patch`
3. Run `run_repro` to verify the bug is fixed
4. Run `run_tests` to verify no regressions
5. If either fails, revise and retry
6. Call `submit_fix` when both pass

Make the MINIMAL change that fixes the bug. Do not refactor.\
"""

class FixAgent(BaseAgent):
    def __init__(self, *, tool_builder: Callable, system_prompt=None, **kwargs):
        kwargs.setdefault("model_id", os.getenv("AUTODEBUG_FIX_MODEL") or os.getenv("AUTODEBUG_MODEL", "claude-sonnet-4-6"))
        kwargs.setdefault("provider", os.getenv("AUTODEBUG_FIX_MODEL_PROVIDER") or os.getenv("AUTODEBUG_MODEL_PROVIDER", "anthropic"))
        super().__init__(**kwargs)
        self._tool_builder = tool_builder
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.repro
        assert state.bisect
        assert state.root_cause

        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))
        patches: list[dict] = []
        result: list = []

        tools = self._tool_builder(
            repo_path=repo_path,
            sandbox=sandbox,
            repro_script=state.repro.repro_script,
            test_command=state.test_command or "",
            patches=patches,
            result=result,
        )
        tool_map = {t.name: t for t in tools}

        test_cmd_note = (
            f"\nTest command to verify fix: `{state.test_command}`\n"
            if state.test_command else ""
        )
        initial_context = (
            f"Root cause:\n{state.root_cause.hypothesis}\n\n"
            f"{state.root_cause.summary}\n\n"
            "Relevant lines:\n"
            + "\n".join(f"  - {l}" for l in state.root_cause.relevant_lines)
            + f"\n\nCulprit commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n"
            + test_cmd_note
            + "\nPlease fix this bug. Start by reading the relevant files."
        )

        for retry in range(self._max_retries + 1):
            budget = self._new_budget()
            result.clear()
            patches.clear()
            messages: list[BaseMessage] = [self.human(initial_context)]
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
                        state.fix = result[0]
                        state.stage = PipelineStage.DONE
                        return state

            except BudgetExceeded:
                if retry < self._max_retries:
                    continue
                state.stage = PipelineStage.FAILED
                state.error = f"FixAgent: budget exceeded after {retry + 1} attempt(s)"
                return state

        state.stage = PipelineStage.FAILED
        state.error = "FixAgent: could not produce a valid fix"
        return state
