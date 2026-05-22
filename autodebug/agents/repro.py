"""Repro Agent — confirms a bug exists and writes a minimal reproducing script."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, ReproResult

MAX_ATTEMPTS = int(os.getenv("REPRO_MAX_ATTEMPTS", "5"))

SYSTEM_PROMPT = """You are an expert software debugger. Your job is to reproduce a bug
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

When you have a confirmed reproducing script, call the `submit_repro` tool."""

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the repository",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory of the repository",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_script",
        "description": "Run a Python script in the sandbox against the repository",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["script"],
        },
    },
    {
        "name": "submit_repro",
        "description": "Submit the confirmed reproduction script when the bug is reproduced",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "The minimal reproducing script"},
                "error_output": {"type": "string", "description": "The error output confirming the bug"},
            },
            "required": ["script", "error_output"],
        },
    },
]


class ReproAgent(BaseAgent):
    """Runs a tool-use loop to write and validate a bug reproduction script."""

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path, "repo must be cloned before repro agent runs"

        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))

        messages: list[BaseMessage] = [
            self.human(
                f"Bug report:\n\n{state.bug_report}\n\n"
                "Please reproduce this bug. Start by exploring the repository structure."
            )
        ]

        for _ in range(MAX_ATTEMPTS):
            response = self._chat(messages, tools=TOOLS, system=SYSTEM_PROMPT)
            state.total_llm_calls += 1
            state.total_tokens += self.count_tokens(response)

            if not response.tool_calls:
                break

            tool_results: list[BaseMessage] = []
            submitted: ReproResult | None = None

            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                tc_id = tc["id"]

                if name == "read_file":
                    file_path = repo_path / args["path"]
                    content = (
                        file_path.read_text(encoding="utf-8", errors="replace")
                        if file_path.exists()
                        else f"File not found: {args['path']}"
                    )

                elif name == "list_files":
                    dir_path = repo_path / args["path"]
                    if dir_path.is_dir():
                        entries = sorted(dir_path.iterdir())
                        content = "\n".join(
                            ("  " if e.is_dir() else "") + e.name for e in entries
                        )
                    else:
                        content = f"Directory not found: {args['path']}"

                elif name == "run_script":
                    run_result = sandbox.run_script(args["script"])
                    content = json.dumps({
                        "exit_code": run_result.exit_code,
                        "stdout": run_result.stdout[-3000:],
                        "stderr": run_result.stderr[-3000:],
                    })

                elif name == "submit_repro":
                    final_run = sandbox.run_script(args["script"])
                    if not final_run.success:
                        submitted = ReproResult(
                            repro_script=args["script"],
                            error_output=args["error_output"],
                            confirmed=True,
                        )
                        content = "Repro confirmed. Pipeline will continue."
                    else:
                        content = (
                            "Script exited with 0 — bug was NOT reproduced. "
                            "Revise the script so it fails when the bug is present."
                        )

                else:
                    content = f"Unknown tool: {name}"

                tool_results.append(self.tool_result(tc_id, content))

            messages.append(response)
            messages.extend(tool_results)

            if submitted:
                state.repro = submitted
                state.stage = PipelineStage.BISECT
                return state

        state.stage = PipelineStage.FAILED
        state.error = f"ReproAgent failed to reproduce the bug after {MAX_ATTEMPTS} attempts"
        return state
