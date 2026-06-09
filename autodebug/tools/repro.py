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


def make_submit_repro_tool(sandbox: Sandbox, result: list, test_command: str = "", **_):
    @tool
    def submit_repro(script: str, error_output: str) -> str:
        """Submit the confirmed reproduction script when the bug is reproduced."""
        final_run = sandbox.run_script(script)
        if final_run.success:
            return "Script exited 0 — bug NOT reproduced. Revise the script so it fails."

        # A non-zero exit alone is weak evidence — an unrelated crash (missing
        # dependency, import error, typo) also exits non-zero. When the benchmark
        # gives us a known failing test, require it to actually fail; that ties the
        # repro to the target bug rather than to an environment artifact.
        if test_command:
            test_run = sandbox.exec(test_command)
            if test_run.success:
                return (
                    f"The script failed, but the known failing test passed "
                    f"(`{test_command}` exited 0), so this isn't reproducing the target "
                    f"bug. Revise the script so that test fails.\n{test_run.output[-2000:]}"
                )

        # Store the *observed* failure from this verifying run, not the agent's
        # claimed `error_output` — downstream stages must see what actually happened.
        result.append(ReproResult(
            repro_script=script,
            error_output=final_run.output[-4000:],
            confirmed=True,
        ))
        return "Repro confirmed. Pipeline will continue."
    return submit_repro
