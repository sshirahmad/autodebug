"""Fix Agent — generates and validates patches using an InspectCoder-style loop."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import BaseMessage

from autodebug.agents.base import BaseAgent
from autodebug.sandbox import Sandbox
from autodebug.state import DebugState, FixResult, PipelineStage

MAX_ATTEMPTS = int(os.getenv("FIX_MAX_ATTEMPTS", "5"))
FIX_MODEL    = os.getenv("AUTODEBUG_FIX_MODEL", "")
FIX_PROVIDER = os.getenv("AUTODEBUG_FIX_MODEL_PROVIDER", "")

SYSTEM_PROMPT = """You are an expert software engineer fixing a confirmed regression bug.

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

Make the MINIMAL change that fixes the bug. Do not refactor."""

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the repository",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply a code change by replacing old content with new content in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "old_content": {"type": "string", "description": "Exact content to replace"},
                "new_content": {"type": "string", "description": "New content to substitute in"},
            },
            "required": ["path", "old_content", "new_content"],
        },
    },
    {
        "name": "run_repro",
        "description": "Run the reproduction script — should PASS after fix is applied",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_tests",
        "description": "Run the test suite to check for regressions",
        "parameters": {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Specific test file or directory (default: all tests)",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_fix",
        "description": "Submit the final validated fix",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence description of what was fixed and how",
                },
            },
            "required": ["summary"],
        },
    },
]


class FixAgent(BaseAgent):
    """Generates and iteratively validates a fix using the repro script and test suite."""

    def __init__(self):
        # Use a stronger model for fix generation if configured separately
        model_id  = FIX_MODEL  or os.getenv("AUTODEBUG_MODEL", "claude-sonnet-4-6")
        provider  = FIX_PROVIDER or os.getenv("AUTODEBUG_MODEL_PROVIDER", "anthropic")
        super().__init__(model_id=model_id, provider=provider)

    def run(self, state: DebugState) -> DebugState:
        assert state.repo_local_path
        assert state.repro
        assert state.bisect
        assert state.root_cause

        repo_path = Path(state.repo_local_path)
        sandbox = Sandbox(str(repo_path))
        applied_patches: list[dict] = []
        attempt = 0

        messages: list[BaseMessage] = [
            self.human(
                f"Root cause:\n{state.root_cause.hypothesis}\n\n"
                f"{state.root_cause.summary}\n\n"
                "Relevant lines:\n"
                + "\n".join(f"  - {l}" for l in state.root_cause.relevant_lines)
                + f"\n\nCulprit commit diff:\n```diff\n{state.bisect.commit_diff}\n```\n\n"
                "Please fix this bug. Start by reading the relevant files."
            )
        ]

        while attempt < MAX_ATTEMPTS:
            response = self._chat(messages, tools=TOOLS, system=SYSTEM_PROMPT)
            state.total_llm_calls += 1
            state.total_tokens += self.count_tokens(response)

            if not response.tool_calls:
                break

            tool_results: list[BaseMessage] = []
            submitted: FixResult | None = None

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

                elif name == "list_files":
                    dpath = repo_path / args["path"]
                    if dpath.is_dir():
                        entries = sorted(dpath.iterdir())
                        content = "\n".join(
                            ("  " if e.is_dir() else "") + e.name for e in entries
                        )
                    else:
                        content = f"Directory not found: {args['path']}"

                elif name == "apply_patch":
                    fpath = repo_path / args["path"]
                    if not fpath.exists():
                        content = f"File not found: {args['path']}"
                    else:
                        original = fpath.read_text(encoding="utf-8")
                        old, new = args["old_content"], args["new_content"]
                        if old not in original:
                            content = (
                                "ERROR: old_content not found verbatim in file. "
                                "Read the file first to get the exact content."
                            )
                        else:
                            fpath.write_text(original.replace(old, new, 1), encoding="utf-8")
                            applied_patches.append({"path": args["path"], "old": old, "new": new})
                            content = f"Patch applied to {args['path']}"
                            attempt += 1

                elif name == "run_repro":
                    run = sandbox.run_script(state.repro.repro_script)
                    content = f"exit_code={run.exit_code}\n{run.output[-3000:]}"

                elif name == "run_tests":
                    test_path = args.get("test_path", "")
                    run = sandbox.run(f"pip install -e . -q && python -m pytest {test_path} -x --tb=short -q")
                    content = f"exit_code={run.exit_code}\n{run.output[-4000:]}"

                elif name == "submit_fix":
                    repro_run = sandbox.run_script(state.repro.repro_script)
                    test_run  = sandbox.run("python -m pytest --tb=short -q")
                    if not repro_run.success:
                        content = f"Cannot submit: repro still fails.\n{repro_run.output[-2000:]}"
                    elif not test_run.success:
                        content = f"Cannot submit: test suite failing.\n{test_run.output[-2000:]}"
                    else:
                        submitted = FixResult(
                            patch=_build_patch_summary(applied_patches),
                            attempts=attempt,
                            test_output=test_run.output[-2000:],
                        )
                        content = "Fix validated and submitted."

                else:
                    content = f"Unknown tool: {name}"

                tool_results.append(self.tool_result(tc["id"], content))

            messages.append(response)
            messages.extend(tool_results)

            if submitted:
                state.fix = submitted
                state.stage = PipelineStage.DONE
                return state

        state.stage = PipelineStage.FAILED
        state.error = f"FixAgent: could not produce a valid fix after {MAX_ATTEMPTS} attempts"
        return state


def _build_patch_summary(patches: list[dict]) -> str:
    if not patches:
        return ""
    lines = []
    for p in patches:
        lines.append(f"--- {p['path']}")
        lines.append(f"+++ {p['path']}")
        for line in p["old"].splitlines():
            lines.append(f"-{line}")
        for line in p["new"].splitlines():
            lines.append(f"+{line}")
        lines.append("")
    return "\n".join(lines)
