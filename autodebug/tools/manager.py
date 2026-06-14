"""Manager tools — expose each existing sub-agent as a tool the Manager calls.

Every tool here wraps one of the plain `run_<name>(state, *, registry)` drivers
(repro/bisect/root_cause/fix). After the sub-agent runs, the tool inspects the
shared `DebugState`, transitions the `FSM`, and returns a concise *signal* the
manager LLM uses to decide what to do next (advance, retry, or loop back for a
fix↔repro/root_cause back-and-forth).

The sub-agent drivers are imported lazily inside each tool so this module can be
imported by autodebug.tools (and discovered by the registry) without pulling in
the autodebug.agents package at import time — avoiding a circular import.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from autodebug.fsm import FSM, ManagerPhase


def _delegate(driver, state, registry) -> str | None:
    """Run a sub-agent driver, swallowing any exception so a single sub-agent
    crash (transient provider error, tool failure, etc.) can't kill the whole
    manager session. Returns an error string on crash, else None."""
    try:
        driver(state, registry=registry)
        return None
    except Exception as exc:  # noqa: BLE001 — resilience boundary
        return f"{type(exc).__name__}: {str(exc)[:300]}"


def make_run_repro_agent_tool(state, registry, fsm: FSM, **_):
    @tool
    def run_repro_agent() -> str:
        """Delegate to the Repro sub-agent: reproduce the bug with a minimal
        failing script. Call this first, or again in REVISING if you suspect the
        existing reproduction is wrong."""
        from autodebug.agents import run_repro

        err = _delegate(run_repro, state, registry)
        if err:
            return (f"REPRO sub-agent crashed: {err}. This may be transient — "
                    "retry run_repro_agent, or finish('failed', ...) if it persists.")
        if state.repro and state.repro.confirmed:
            # Re-reproducing during REVISING keeps us in REVISING; otherwise advance.
            if fsm.phase != ManagerPhase.REVISING:
                fsm.to(ManagerPhase.REPRODUCED)
            return (
                "REPRO CONFIRMED — the bug reproduces. Observed failure:\n"
                f"{state.repro.error_output[:1500]}"
            )
        return (
            "REPRO FAILED — could not produce a failing reproduction. "
            "You may call run_repro_agent again or finish('failed', ...)."
        )

    return run_repro_agent


def make_run_bisect_agent_tool(state, registry, fsm: FSM, **_):
    @tool
    def run_bisect_agent() -> str:
        """Delegate to the Bisect sub-agent: find the commit that introduced the
        bug. Requires a confirmed reproduction."""
        from autodebug.agents import run_bisect

        if not (state.repro and state.repro.confirmed):
            return "Cannot bisect yet — reproduce the bug first (run_repro_agent)."
        err = _delegate(run_bisect, state, registry)
        if err:
            return (f"BISECT sub-agent crashed: {err}. This may be transient — "
                    "retry run_bisect_agent, or finish('failed', ...) if it persists.")
        # Defense in depth: a result with a blank culprit SHA is not a real
        # find. Treat it as a failure (and clear it) so the FSM doesn't advance
        # on a bogus culprit and a retry starts clean.
        if state.bisect and state.bisect.culprit_commit.strip():
            fsm.to(ManagerPhase.BISECTED)
            return (
                f"CULPRIT FOUND: {state.bisect.culprit_commit} — "
                f"{state.bisect.commit_message}"
            )
        state.bisect = None
        return "BISECT FAILED — no culprit commit identified. Retry or finish('failed', ...)."

    return run_bisect_agent


def make_run_root_cause_agent_tool(state, registry, fsm: FSM, **_):
    @tool
    def run_root_cause_agent() -> str:
        """Delegate to the Root-Cause sub-agent: explain WHY the culprit broke
        things. Requires a culprit commit. Re-run in REVISING to refine a
        hypothesis that led to a failed fix."""
        from autodebug.agents import run_root_cause

        if not state.bisect:
            return "Cannot analyze yet — find the culprit first (run_bisect_agent)."
        err = _delegate(run_root_cause, state, registry)
        if err:
            return (f"ROOT CAUSE sub-agent crashed: {err}. This may be transient — "
                    "retry run_root_cause_agent, or finish('failed', ...) if it persists.")
        if state.root_cause:
            fsm.to(ManagerPhase.ANALYZED)
            return f"ROOT CAUSE: {state.root_cause.hypothesis}"
        return "ROOT CAUSE FAILED — no hypothesis produced. Retry or finish('failed', ...)."

    return run_root_cause_agent


def make_run_fix_agent_tool(state, registry, fsm: FSM, **_):
    @tool
    def run_fix_agent() -> str:
        """Delegate to the Fix sub-agent: write a patch and verify the repro +
        tests pass. Requires a root cause. On failure you enter REVISING and may
        loop back to repro/root_cause before retrying."""
        from autodebug.agents import run_fix

        if not state.root_cause:
            return "Cannot fix yet — determine the root cause first (run_root_cause_agent)."
        err = _delegate(run_fix, state, registry)
        if err:
            return (f"FIX sub-agent crashed: {err}. This may be transient — "
                    "retry run_fix_agent, or finish('failed', ...) if it persists.")
        if state.fix:
            fsm.to(ManagerPhase.DONE)
            return (
                "FIX VERIFIED — the reproduction passes and targeted tests pass. "
                "Call finish('success', <one-line summary>) to conclude."
            )
        fsm.to(ManagerPhase.REVISING)
        return (
            "FIX FAILED — the patch did not make the repro/tests pass. Options:\n"
            "  - run_repro_agent: the reproduction may be wrong or imprecise\n"
            "  - run_root_cause_agent: refine the hypothesis\n"
            "  - run_fix_agent: try a different patch\n"
            "If you've exhausted reasonable options, call finish('failed', <why>)."
        )

    return run_fix_agent


def make_finish_tool(state, fsm: FSM, **_):
    @tool(parse_docstring=True)
    def finish(
        outcome: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        summary: str = "",
    ) -> Command:
        """Conclude the debugging session.

        Args:
            outcome: 'success' (a fix was verified) or 'failed' (giving up).
            summary: one-line explanation of the outcome.
        """
        ok = outcome.strip().lower() == "success"
        fsm.to(ManagerPhase.DONE if ok else ManagerPhase.FAILED)
        return Command(update={
            "outcome": {"outcome": "success" if ok else "failed", "summary": summary},
            "messages": [ToolMessage("Session concluded.", tool_call_id=tool_call_id)],
        })

    return finish
