"""Tools for the Repro agent: run_script, submit_repro."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from autodebug.sandbox import Sandbox
from autodebug.state import ReproResult


def make_run_script_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def run_script(script: str) -> str:
        """Run a Python script in the sandbox against the repository.

        Args:
            script: The Python source to execute (cwd is the repo root).
        """
        result = sandbox.run_script(script)
        return json.dumps({
            "exit_code": result.exit_code,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-3000:],
        })
    return run_script


def make_submit_repro_tool(sandbox: Sandbox, result: list, **_):
    @tool(parse_docstring=True)
    def submit_repro(script: str, error_output: str) -> str:
        """Submit the confirmed reproduction script when the bug is reproduced.

        Args:
            script: The minimal Python script that reproduces the bug. It must
                exit non-zero and the failure must match the reported symptom
                (not an unrelated import/dependency error in the sandbox).
            error_output: The failure output you observed running the script.
        """
        final_run = sandbox.run_script(script)
        if final_run.success:
            return "Script exited 0 — bug NOT reproduced. Revise the script so it fails."

        # The reproduction IS the oracle for the rest of the pipeline (there is no
        # benchmark test in production). Store the *observed* failure from this
        # verifying run, not the agent's claimed string, so downstream stages and
        # the fixer see what actually happened.
        result.append(ReproResult(
            repro_script=script,
            error_output=final_run.output[-4000:],
            confirmed=True,
        ))
        return "Repro confirmed. Pipeline will continue."
    return submit_repro
