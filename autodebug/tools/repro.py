"""Tools for the Repro agent: run_script, submit_repro."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from autodebug.sandbox import Sandbox
from autodebug.state import ReproResult


def make_run_script_tool(sandbox: Sandbox, **_):
    @tool
    def run_script(script: str) -> str:
        """Run a Python script in the sandbox against the repository."""
        result = sandbox.run_script(script)
        return json.dumps({
            "exit_code": result.exit_code,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-3000:],
        })
    return run_script


def make_submit_repro_tool(sandbox: Sandbox, result: list, **_):
    @tool
    def submit_repro(script: str, error_output: str) -> str:
        """Submit the confirmed reproduction script when the bug is reproduced."""
        final_run = sandbox.run_script(script)
        if not final_run.success:
            result.append(ReproResult(repro_script=script, error_output=error_output, confirmed=True))
            return "Repro confirmed. Pipeline will continue."
        return "Script exited 0 — bug NOT reproduced. Revise the script so it fails."
    return submit_repro
