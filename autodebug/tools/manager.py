"""Manager tools — expose each existing sub-agent as a tool the Manager calls.

Every tool wraps one of the plain `run_<name>(state, *, registry)` drivers
(repro/bisect/root_cause/fix). The Manager runs as a *checkpointed* agent: the
pipeline state and FSM phase live in the graph state (``debug`` / ``fsm_phase``
channels of ManagerAgentState), not external closures, so the Manager can be a
served, resumable subgraph. Each tool therefore:

  1. reads the current ``debug`` (a DebugState dump) + ``fsm_phase`` from the
     injected graph state,
  2. runs the sub-agent driver (which mutates a rebuilt DebugState),
  3. returns a ``Command`` that writes the updated ``debug``/``fsm_phase`` back and
     a concise *signal* ToolMessage the manager LLM uses to decide what to do next.

The pure ``_*_step`` functions hold the transition logic and are unit-tested
directly; the ``@tool`` wrappers are thin state adapters. Sub-agent drivers are
imported lazily so this module stays importable without the heavy agents package.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from autodebug.fsm import ManagerPhase
from autodebug.state import DebugState

logger = logging.getLogger(__name__)


def _delegate(driver, state, registry) -> str | None:
    """Run a sub-agent driver, swallowing any exception so a single sub-agent
    crash (transient provider error, tool failure, etc.) can't kill the whole
    manager session. Returns an error string on crash, else None."""
    try:
        driver(state, registry=registry)
        return None
    except Exception as exc:  # noqa: BLE001 — resilience boundary
        logger.warning("Sub-agent %s crashed: %s: %s",
                       getattr(driver, "__name__", driver), type(exc).__name__, exc)
        return f"{type(exc).__name__}: {str(exc)[:300]}"


def _load(state) -> DebugState:
    return DebugState(**((state or {}).get("debug") or {}))


def _phase(state) -> str:
    return (state or {}).get("fsm_phase") or ManagerPhase.INIT.value


def _command(ds: DebugState, new_phase: Optional[ManagerPhase], signal: str,
             tool_call_id: str) -> Command:
    """Write the updated pipeline state + (optional) phase back, plus the signal."""
    update: dict = {
        "debug": ds.model_dump(),
        "messages": [ToolMessage(signal, tool_call_id=tool_call_id)],
    }
    if new_phase is not None:
        update["fsm_phase"] = new_phase.value
    return Command(update=update)


# ---------------------------------------------------------------------------
# Pure step functions: mutate `ds` via the driver, return (new_phase|None, signal).
# `new_phase is None` means "stay in the current phase".
# ---------------------------------------------------------------------------

def _repro_step(ds: DebugState, phase: str, registry) -> tuple[Optional[ManagerPhase], str]:
    from autodebug.agents import run_repro

    err = _delegate(run_repro, ds, registry)
    if err:
        return None, (f"REPRO sub-agent crashed: {err}. This may be transient — "
                      "retry run_repro_agent, or finish('failed', ...) if it persists.")
    if ds.repro and ds.repro.confirmed:
        # Re-reproducing during REVISING keeps us in REVISING; otherwise advance.
        new = None if phase == ManagerPhase.REVISING.value else ManagerPhase.REPRODUCED
        return new, ("REPRO CONFIRMED — the bug reproduces. Observed failure:\n"
                     f"{ds.repro.error_output[:1500]}")
    return None, ("REPRO FAILED — could not produce a failing reproduction. "
                  "You may call run_repro_agent again or finish('failed', ...).")


def _bisect_step(ds: DebugState, phase: str, registry) -> tuple[Optional[ManagerPhase], str]:
    from autodebug.agents import run_bisect

    if not (ds.repro and ds.repro.confirmed):
        return None, "Cannot bisect yet — reproduce the bug first (run_repro_agent)."
    err = _delegate(run_bisect, ds, registry)
    if err:
        return None, (f"BISECT sub-agent crashed: {err}. This may be transient — "
                      "retry run_bisect_agent, or finish('failed', ...) if it persists.")
    # Defense in depth: a blank culprit SHA is not a real find. Treat it as a
    # failure (and clear it) so the FSM doesn't advance on a bogus culprit.
    if ds.bisect and ds.bisect.culprit_commit.strip():
        return ManagerPhase.BISECTED, (f"CULPRIT FOUND: {ds.bisect.culprit_commit} — "
                                       f"{ds.bisect.commit_message}")
    ds.bisect = None
    return None, ("BISECT INCONCLUSIVE — no culprit commit identified. The culprit is "
                  "helpful but NOT required: call run_root_cause_agent to analyze from the "
                  "reproduction and bug report directly (it just won't have a culprit "
                  "diff). Retry bisect only if you have a concrete new idea.")


def _root_cause_step(ds: DebugState, phase: str, registry) -> tuple[Optional[ManagerPhase], str]:
    from autodebug.agents import run_root_cause

    if not (ds.repro and ds.repro.confirmed):
        return None, "Cannot analyze yet — reproduce the bug first (run_repro_agent)."
    err = _delegate(run_root_cause, ds, registry)
    if err:
        return None, (f"ROOT CAUSE sub-agent crashed: {err}. This may be transient — "
                      "retry run_root_cause_agent, or finish('failed', ...) if it persists.")
    if ds.root_cause:
        return ManagerPhase.ANALYZED, f"ROOT CAUSE: {ds.root_cause.hypothesis}"
    return None, "ROOT CAUSE FAILED — no hypothesis produced. Retry or finish('failed', ...)."


def _fix_step(ds: DebugState, phase: str, registry) -> tuple[Optional[ManagerPhase], str]:
    from autodebug.agents import run_fix

    if not ds.root_cause:
        return None, "Cannot fix yet — determine the root cause first (run_root_cause_agent)."
    err = _delegate(run_fix, ds, registry)
    if err:
        return None, (f"FIX sub-agent crashed: {err}. This may be transient — "
                      "retry run_fix_agent, or finish('failed', ...) if it persists.")
    if ds.fix:
        return ManagerPhase.DONE, ("FIX VERIFIED — the reproduction passes and targeted "
                                   "tests pass. Call finish('success', <one-line summary>).")
    return ManagerPhase.REVISING, (
        "FIX FAILED — the patch did not make the repro/tests pass. Options:\n"
        "  - run_repro_agent: the reproduction may be wrong or imprecise\n"
        "  - run_root_cause_agent: refine the hypothesis\n"
        "  - run_fix_agent: try a different patch\n"
        "If you've exhausted reasonable options, call finish('failed', <why>).")


# ---------------------------------------------------------------------------
# Tool wrappers: thin adapters from injected graph state -> step -> Command.
# ---------------------------------------------------------------------------

def make_run_repro_agent_tool(registry=None, **_):
    @tool
    def run_repro_agent(state: Annotated[dict, InjectedState],
                        tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Delegate to the Repro sub-agent: reproduce the bug with a minimal
        failing script. Call this first, or again in REVISING if you suspect the
        existing reproduction is wrong."""
        ds = _load(state)
        new_phase, signal = _repro_step(ds, _phase(state), registry)
        return _command(ds, new_phase, signal, tool_call_id)

    return run_repro_agent


def make_run_bisect_agent_tool(registry=None, **_):
    @tool
    def run_bisect_agent(state: Annotated[dict, InjectedState],
                         tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Delegate to the Bisect sub-agent: find the commit that introduced the
        bug. Requires a confirmed reproduction."""
        ds = _load(state)
        new_phase, signal = _bisect_step(ds, _phase(state), registry)
        return _command(ds, new_phase, signal, tool_call_id)

    return run_bisect_agent


def make_run_root_cause_agent_tool(registry=None, **_):
    @tool
    def run_root_cause_agent(state: Annotated[dict, InjectedState],
                             tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Delegate to the Root-Cause sub-agent: explain WHY the bug happens.
        Uses the culprit diff if one was found, but works from the reproduction +
        bug report alone if bisect was inconclusive. Re-run in REVISING to refine a
        hypothesis that led to a failed fix."""
        ds = _load(state)
        new_phase, signal = _root_cause_step(ds, _phase(state), registry)
        return _command(ds, new_phase, signal, tool_call_id)

    return run_root_cause_agent


def make_run_fix_agent_tool(registry=None, **_):
    @tool
    def run_fix_agent(state: Annotated[dict, InjectedState],
                      tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Delegate to the Fix sub-agent: write a patch and verify the repro +
        tests pass. Requires a root cause. On failure you enter REVISING and may
        loop back to repro/root_cause before retrying."""
        ds = _load(state)
        new_phase, signal = _fix_step(ds, _phase(state), registry)
        return _command(ds, new_phase, signal, tool_call_id)

    return run_fix_agent


def make_finish_tool(registry=None, **_):
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
        phase = ManagerPhase.DONE if ok else ManagerPhase.FAILED
        return Command(update={
            "outcome": {"outcome": "success" if ok else "failed", "summary": summary},
            "fsm_phase": phase.value,
            "messages": [ToolMessage("Session concluded.", tool_call_id=tool_call_id)],
        })

    return finish
