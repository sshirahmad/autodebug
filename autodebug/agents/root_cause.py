"""Root Cause Agent — explains WHY the culprit commit broke things."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, PipelineStage, RootCauseResult
from autodebug.tools import git_utils

SYSTEM_PROMPT = """You are an expert software engineer performing a root cause analysis.

You have:
1. A commit diff that introduced a regression
2. The output of running the reproduction script
3. Access to the repository files for deeper investigation

Your job: produce a precise root cause analysis that explains:
- What changed in the culprit commit
- Why that change caused the bug
- Which specific lines are responsible

Use the tools to read any files you need for context.
When you have enough information, call `submit_root_cause`."""

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the repository at HEAD",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file_at_parent",
        "description": "Read a file as it was BEFORE the culprit commit (to compare)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_repro_with_traceback",
        "description": "Run the reproduction script and capture the full Python traceback",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_root_cause",
        "description": "Submit the root cause analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One paragraph summary of the root cause",
                },
                "relevant_lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 'file.py:line_number — explanation' strings",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "The core hypothesis: 'The bug is caused by X because Y'",
                },
            },
            "required": ["summary", "relevant_lines", "hypothesis"],
        },
    },
]


class RootCauseAgent(BaseAgent):
    """Analyzes the culprit commit diff + runtime output to explain the root cause."""

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.bisect
        assert state.repro

        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))
        parent_sha = f"{state.bisect.culprit_commit}^"

        initial_run = sandbox.run_script(state.repro.repro_script)

        messages: list[BaseMessage] = [
            self.human(
                f"Culprit commit: {state.bisect.commit_message}\n"
                f"SHA: {state.bisect.culprit_commit}\n\n"
                f"Commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
                f"Reproduction script output:\n```\n{initial_run.output[-3000:]}\n```\n\n"
                "Please identify the root cause of this regression."
            )
        ]

        for _ in range(5):
            response = self._chat(messages, tools=TOOLS, system=SYSTEM_PROMPT)
            state.total_llm_calls += 1
            state.total_tokens += self.count_tokens(response)

            if not response.tool_calls:
                break

            tool_results: list[BaseMessage] = []
            submitted: RootCauseResult | None = None

            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]

                if name == "read_file":
                    fpath = repo_path / args["path"]
                    content = (
                        fpath.read_text(encoding="utf-8", errors="replace")
                        if fpath.exists()
                        else f"File not found: {args['path']}"
                    )

                elif name == "read_file_at_parent":
                    content = git_utils.get_file_at_commit(
                        str(repo_path), parent_sha, args["path"]
                    ) or f"File not found at parent commit: {args['path']}"

                elif name == "run_repro_with_traceback":
                    run = sandbox.run_script(
                        "import traceback, sys\n"
                        "try:\n"
                        "    exec(open('/workspace/repro.py').read())\n"
                        "except Exception:\n"
                        "    traceback.print_exc()\n"
                        "    sys.exit(1)\n",
                        extra_files={"repro.py": state.repro.repro_script},
                    )
                    content = run.output[-4000:]

                elif name == "submit_root_cause":
                    submitted = RootCauseResult(
                        summary=args["summary"],
                        relevant_lines=args["relevant_lines"],
                        hypothesis=args["hypothesis"],
                    )
                    content = "Root cause submitted."

                else:
                    content = f"Unknown tool: {name}"

                tool_results.append(self.tool_result(tc["id"], content))

            messages.append(response)
            messages.extend(tool_results)

            if submitted:
                state.root_cause = submitted
                state.stage = PipelineStage.FIX
                return state

        state.stage = PipelineStage.FAILED
        state.error = "RootCauseAgent: could not determine root cause"
        return state
