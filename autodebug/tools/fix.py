"""Tools for the Fix agent."""

from __future__ import annotations

import base64
import os
import re
import shlex
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from autodebug.patch_utils import is_test_path
from autodebug.sandbox import REPO_DIR, Sandbox
from autodebug.state import FixResult
from autodebug.tools.introspect import postmortem_harness

# pytest's short-summary lines for failures/errors look like
# "FAILED path::test - reason" / "ERROR path - reason"; grab the node id.
_PYTEST_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _collect_failures(sandbox: Sandbox, targets: str) -> set[str]:
    """Run the given pytest targets and return the set of failing/erroring node IDs."""
    run = sandbox.exec(
        f"python -m pytest {targets} -p no:cacheprovider --tb=no -q -rfE"
    )
    return set(_PYTEST_FAIL_RE.findall(run.output))


def _discover_regression_tests(sandbox: Sandbox) -> str:
    """Find existing test files covering the source files this fix changed.

    Maps each changed source module `foo.py` to `test_foo.py` / `foo_test.py`
    anywhere in the repo (pytest naming convention). Returns a space-separated,
    shell-quoted target string (capped), or "" if none are found. These are the
    project's OWN tests at the buggy commit — never the hidden benchmark test.
    """
    changed = sandbox.git("diff", "--name-only").stdout.splitlines()
    bases = {
        os.path.splitext(os.path.basename(p))[0]
        for p in changed
        if p.strip() and p.endswith(".py") and not is_test_path(p)
    }
    bases = {b for b in bases if re.fullmatch(r"[\w.-]+", b)}
    found: list[str] = []
    seen: set[str] = set()
    for base in sorted(bases):
        r = sandbox.exec(
            f"find {REPO_DIR} -type f "
            f"\\( -name 'test_{base}.py' -o -name '{base}_test.py' \\) "
            "-not -path '*/.git/*'"
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                found.append(line)
    return " ".join(shlex.quote(p) for p in found[:5])


def make_apply_patch_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def apply_patch(
        path: str, old_content: str, new_content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command | str:
        """Apply a code change by replacing old content with new content in a file.

        Args:
            path: Relative path from the repo root to the file to edit (not a test file).
            old_content: Exact text to replace — must match the file verbatim,
                whitespace included.
            new_content: Replacement text.
        """
        # Never let the fix touch test files: the targeted test is the success
        # oracle (submit_fix runs it), so editing it would let the agent "pass"
        # by changing the test instead of fixing the source.
        if is_test_path(path):
            return (
                f"Refused: '{path}' is a test file. Patch the source code, not the "
                "test — the test is the success criterion and must stay unchanged."
            )
        full = f"{REPO_DIR}/{path.lstrip('/')}"
        check = sandbox.exec(f"test -f {shlex.quote(full)}")
        if check.exit_code != 0:
            return f"File not found: {path}"
        original = sandbox.exec(f"cat {shlex.quote(full)}").stdout
        if old_content not in original:
            return (
                "ERROR: old_content not found verbatim in file. "
                "Read the file first to get exact content."
            )
        updated = original.replace(old_content, new_content, 1)
        sandbox.write_file(path, updated)
        # `patches` has an `add` reducer — this record is appended to the running list.
        return Command(update={
            "patches": [{"path": path, "old": old_content, "new": new_content}],
            "messages": [ToolMessage(f"Patch applied to {path}", tool_call_id=tool_call_id)],
        })
    return apply_patch


def make_run_repro_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool
    def run_repro() -> str:
        """Run the reproduction script — should PASS (exit 0) after the fix.

        On failure it returns the full traceback and the local variables at each
        frame, so you can see *why* the patch didn't work instead of guessing.
        """
        run = sandbox.run_script(postmortem_harness(repro_script))
        if run.success:
            return "exit_code=0 — reproduction PASSES (no exception). The fix works."
        # keep the head: exception + innermost frames (where the patch went wrong)
        return f"exit_code={run.exit_code} — reproduction still FAILS:\n{run.output[:3500]}"
    return run_repro


def make_run_tests_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def run_tests(test_path: str = "") -> str:
        """Run the project's own tests with pytest to check for regressions.

        Args:
            test_path: Path to a test file or directory to run (relative to the
                repo root). Omit to run the whole suite (may be slow).
        """
        cmd = f"python -m pytest {test_path} -x --tb=short -q"
        run = sandbox.exec(cmd)
        return f"exit_code={run.exit_code}\n{run.output[-4000:]}"
    return run_tests


def make_submit_fix_tool(sandbox: Sandbox, repro_script: str, **_):
    @tool(parse_docstring=True)
    def submit_fix(
        summary: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        regression_tests: str = "",
    ) -> Command | str:
        """Submit the fix once the reproduction passes.

        The reproduction is the primary success criterion (there is no benchmark
        test in production). Before accepting, this also runs the project's own
        existing tests for the files you changed and REJECTS the fix if it makes a
        test fail that passed on the buggy code (a regression) — so an over-eager
        patch that breaks neighbouring behavior is caught here.

        Args:
            summary: One-line description of the fix you applied.
            regression_tests: Optional space-separated test path(s) to use for the
                regression check. Leave empty to auto-discover the test files for
                the modules you changed.
        """
        repro_run = sandbox.run_script(repro_script)
        if not repro_run.success:
            return f"Cannot submit: the reproduction still fails.\n{repro_run.output[-2000:]}"
        # Capture the actual source changes as a real, applicable diff so the fix
        # can be validated independently later — not just a textual summary.
        diff = sandbox.git("diff").stdout
        if not diff.strip():
            return (
                "Cannot submit: there are no source changes (empty diff). The "
                "reproduction passing without a code change means it isn't testing "
                "the bug — revise the fix (or the reproduction) before submitting."
            )
        # git apply requires a trailing newline; normalize so the patch is valid.
        diff = diff.rstrip("\n") + "\n"

        # Regression gate: a fix that passes the reproduction can still break
        # adjacent behavior. Run the changed modules' existing tests and block
        # only on tests that REGRESS (passed on the buggy code, fail after the
        # patch). Baselining means a test already failing/flaky at the buggy
        # commit never causes a false rejection.
        targets = regression_tests.strip() or _discover_regression_tests(sandbox)
        gate_note = " No existing test module was found to regression-check."
        if targets:
            stash = sandbox.git("stash")
            if stash.exit_code != 0 or "No local changes" in stash.output:
                # Couldn't isolate the patch to baseline — skip rather than risk a
                # false regression (a pre-existing failure would look like one).
                gate_note = " Regression check skipped (could not baseline)."
            else:
                baseline = _collect_failures(sandbox, targets)
                if sandbox.git("stash", "pop").exit_code != 0:
                    # Restore the fix from the captured diff so state stays intact.
                    enc = base64.b64encode(diff.encode()).decode("ascii")
                    sandbox.exec(f"echo {enc} | base64 -d | git apply --whitespace=nowarn -")
                post = _collect_failures(sandbox, targets)
                regressions = sorted(post - baseline)
                if regressions:
                    return (
                        "Cannot submit: your fix makes tests FAIL that passed on the "
                        "buggy code (regressions):\n  " + "\n  ".join(regressions) + "\n"
                        "Adjust the patch so these keep passing. If one of them encodes "
                        "the OLD buggy behavior, re-check that your fix is correct."
                    )
                gate_note = (
                    f" Regression check passed on {len(targets.split())} existing "
                    "test file(s)."
                )

        # `attempts` is filled in by the driver from the patches channel.
        fix = FixResult(patch=diff, attempts=0, test_output=targets)
        return Command(update={
            "fix": fix.model_dump(),
            "messages": [ToolMessage(
                "Fix validated against the reproduction and submitted." + gate_note,
                tool_call_id=tool_call_id)],
        })
    return submit_fix


def _build_patch_summary(patches: list[dict]) -> str:
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
