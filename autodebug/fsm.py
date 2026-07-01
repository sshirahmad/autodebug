"""Finite-state machine for the Manager agent + FSM-driven middleware.

The Manager ("brain") delegates to the existing sub-agents (repro, bisect,
root_cause, fix). Its behaviour is governed by an explicit FSM:

  * which sub-agents it may call               -> `fsm_tool_gate`  (wrap_model_call)
  * which system prompt it sees                -> `fsm_prompt`      (dynamic_prompt)

The FSM phase is advanced by the sub-agent tools themselves (see
autodebug/tools/manager.py): each tool runs a sub-agent and, based on the
*signal* it gets back (confirmed / failed / fix-verified / ...), transitions
the shared `FSM` instance. The middleware below simply reflects the current
phase into the next model call.

This module lives at the top level (not under autodebug.agents) so that
autodebug.tools can import it without triggering the autodebug.agents package
import — which would create a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from langchain.agents.middleware import dynamic_prompt, wrap_model_call, wrap_tool_call


class ManagerPhase(str, Enum):
    """States of the Manager FSM.

    The happy path is INIT -> REPRODUCED -> BISECTED -> ANALYZED -> DONE.
    A failed fix drops into REVISING, where the manager may loop back to any
    earlier sub-agent (repro/root_cause/fix) before retrying or giving up.
    """

    INIT = "init"
    REPRODUCED = "reproduced"
    BISECTED = "bisected"
    ANALYZED = "analyzed"
    REVISING = "revising"
    DONE = "done"
    FAILED = "failed"


# Terminal phases: once here, the manager loop should end.
TERMINAL: frozenset[ManagerPhase] = frozenset({ManagerPhase.DONE, ManagerPhase.FAILED})


# Which sub-agent tools the Manager may call in each phase — the FSM's transition
# guard. The tool-gate hides every other tool from the model on a given turn,
# steering repro -> bisect -> root_cause -> fix, while REVISING re-opens the
# earlier agents so a failed fix can loop back (fix <-> repro/root_cause).
# `finish` is allowed in EVERY phase: it's the universal escape hatch. Without it
# the manager can be trapped with no legal move — if every sub-agent tool for the
# phase has hit its tool_call_limit, the agent has nothing left to call and just
# narrates "all tools are exhausted", looping until the budget trips.
MANAGER_ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    ManagerPhase.INIT.value:       ("run_repro_agent", "finish"),
    # root_cause is reachable from REPRODUCED too: bisect is best-effort, so if it
    # can't pin a culprit the manager can still analyze from the repro + report.
    ManagerPhase.REPRODUCED.value: (
        "run_bisect_agent", "run_root_cause_agent", "run_repro_agent", "finish",
    ),
    ManagerPhase.BISECTED.value:   ("run_root_cause_agent", "run_repro_agent", "finish"),
    ManagerPhase.ANALYZED.value:   ("run_fix_agent", "run_root_cause_agent", "finish"),
    ManagerPhase.REVISING.value:   (
        "run_repro_agent", "run_root_cause_agent", "run_fix_agent", "finish",
    ),
}


@dataclass
class FSM:
    """Mutable holder for the manager's current phase + transition history."""

    phase: ManagerPhase = ManagerPhase.INIT
    history: list[ManagerPhase] = field(default_factory=list)

    def to(self, phase: ManagerPhase) -> ManagerPhase:
        """Transition to `phase`, recording where we came from."""
        self.history.append(self.phase)
        self.phase = phase
        return phase

    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL


# ----------------------------------------------------------------------
# Pure selection logic (unit-testable without the middleware machinery)
# ----------------------------------------------------------------------


def select_prompt(fsm: FSM, prompts: dict, progress_fn=None) -> str:
    """Return the system prompt for the current phase, plus optional progress."""
    base = prompts.get(fsm.phase) or prompts.get(ManagerPhase.INIT.value) or ""
    if progress_fn is not None:
        return f"{base}\n\n{progress_fn()}".strip()
    return base


# Infrastructure tools injected by middleware (e.g. TodoListMiddleware's
# write_todos) are not part of the FSM and must survive phase gating.
_ALWAYS_ALLOWED: frozenset[str] = frozenset({"write_todos"})


def allowed_tool_names(fsm: FSM, allowed: dict) -> set | None:
    """The set of tool names permitted in the current phase (incl. always-allowed),
    or None when the phase imposes no restriction."""
    names = allowed.get(fsm.phase)
    if names is None:
        return None
    return set(names) | _ALWAYS_ALLOWED


def filter_tools(fsm: FSM, tools: list, allowed: dict) -> list:
    """Return only the tools whose name is allowed in the current phase.

    A phase missing from `allowed` (or mapped to None) leaves the list as-is.
    Infrastructure tools in `_ALWAYS_ALLOWED` are kept regardless of phase.
    """
    names = allowed_tool_names(fsm, allowed)
    if names is None:
        return tools
    return [t for t in tools if getattr(t, "name", None) in names]


def _tool_call_field(tool_call, key: str):
    """Read a field from a tool_call that may be a dict or an object."""
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


# ----------------------------------------------------------------------
# Middleware factories (thin wrappers over the pure logic above)
# ----------------------------------------------------------------------


def _phase_from_state(state) -> FSM:
    """Reconstruct a transient FSM from the checkpointed ``fsm_phase`` channel.

    The phase lives in the graph state (not a closure) so the Manager can run as a
    served, resumable subgraph; the middleware below build a throwaway FSM each call
    purely to reuse the pure selection helpers above."""
    phase = (state or {}).get("fsm_phase") or ManagerPhase.INIT.value
    try:
        return FSM(phase=ManagerPhase(phase))
    except ValueError:
        return FSM()


def fsm_prompt(prompts: dict, progress_fn=None):
    """`dynamic_prompt` middleware: pick the system prompt for the state's phase.

    `prompts` maps phase value (str) -> system prompt. `progress_fn`, if given, is
    called with the graph state and its return value is appended so the manager
    always sees a compact, up-to-date status block.
    """

    @dynamic_prompt
    def _prompt(request) -> str:
        fsm = _phase_from_state(request.state)
        pf = (lambda: progress_fn(request.state)) if progress_fn else None
        return select_prompt(fsm, prompts, pf)

    return _prompt


def fsm_tool_gate(allowed: dict):
    """`wrap_model_call` middleware: expose only the tools allowed in this phase.

    The tools stay registered with the agent — this only controls what the model
    is offered on a given turn, which is how we steer the FSM.
    """

    @wrap_model_call
    def _gate(request, handler):
        request.tools = filter_tools(_phase_from_state(request.state), request.tools, allowed)
        return handler(request)

    return _gate


def fsm_tool_enforce(allowed: dict):
    """`wrap_tool_call` middleware: *enforce* the phase gate at execution time.

    `fsm_tool_gate` only controls which tools are advertised to the model — the
    agent's tool node is still built with the full toolset, so a tool call that
    slips through (hallucinated name, cached history, parallel calls) would still
    run. This hook intercepts every tool execution and, if the tool isn't allowed
    in the current phase, short-circuits with an error ToolMessage instead of
    running it. The model receives that as the tool's result and retries with a
    permitted tool — and because we return a real ToolMessage, the conversation
    stays valid (no dangling tool_use).
    """
    from langchain_core.messages import ToolMessage

    @wrap_tool_call
    def _enforce(request, handler):
        fsm = _phase_from_state(request.state)
        names = allowed_tool_names(fsm, allowed)
        if names is not None:
            tool_name = _tool_call_field(request.tool_call, "name")
            if tool_name not in names:
                return ToolMessage(
                    content=(
                        f"Tool '{tool_name}' is not available in phase "
                        f"'{fsm.phase}'. Allowed now: {sorted(names)}. "
                        f"Call one of those instead."
                    ),
                    tool_call_id=_tool_call_field(request.tool_call, "id"),
                    name=tool_name,
                    status="error",
                )
        return handler(request)

    return _enforce


# Blocking stages whose retry-exhaustion should pause for a human (NOT optional bisect).
_HITL_STAGES = ("run_repro_agent", "run_root_cause_agent", "run_fix_agent")
_HITL_FAIL_MARKERS = ("FAILED", "crashed")


def _hitl_recap(ds) -> str:
    bits = []
    if ds.repro:
        bits.append(f"repro {'confirmed' if ds.repro.confirmed else 'unconfirmed'}")
    if ds.bisect and ds.bisect.culprit_commit.strip():
        bits.append(f"culprit {ds.bisect.culprit_commit[:8]}")
    if ds.root_cause:
        bits.append(f"root cause: {ds.root_cause.summary[:100]}")
    if ds.hypothesis_attempts:
        bits.append(f"{len(ds.hypothesis_attempts)} fix attempt(s)")
    return "; ".join(bits) or "no artifacts yet"


def stage_hitl_middleware(stages: tuple[str, ...] = _HITL_STAGES):
    """`wrap_tool_call` middleware (served/interactive build ONLY): when a blocking
    sub-agent tool reports failure — its retries exhausted — pause via ``interrupt()``
    with a summary of what's been done, then fold the developer's reply into the tool
    result so the Manager continues *in the same conversation*, steered by it.

    This is added only to the served Manager (eval/CLI-run_manager omit it), so a
    stage failure never blocks on input that won't come. Because the Manager runs as a
    subgraph node, the interrupt bubbles to the served stream (Studio / Agent Chat UI).
    """
    from langgraph.types import Command, interrupt

    from autodebug.state import DebugState

    blocking = set(stages)

    @wrap_tool_call
    def _hitl(request, handler):
        result = handler(request)
        name = _tool_call_field(request.tool_call, "name")
        if name not in blocking or not isinstance(result, Command):
            return result
        msgs = (result.update or {}).get("messages") if isinstance(result.update, dict) else None
        tm = msgs[0] if msgs else None
        signal = getattr(tm, "content", "") if tm is not None else ""
        if not any(marker in signal for marker in _HITL_FAIL_MARKERS):
            return result  # the stage succeeded (or advanced) — nothing to ask

        ds = DebugState(**((request.state or {}).get("debug") or {"repo_url": "", "bug_report": ""}))
        summary = (
            f"Stage `{name}` exhausted its retries without success.\n\n"
            f"Progress so far: {_hitl_recap(ds)}\n\nSignal:\n{signal}\n\n"
            "Reply with guidance to steer the next attempt (a file/function to focus on, a "
            "hypothesis to try, a different repro approach), or 'skip' to let it proceed."
        )
        feedback = interrupt({"type": "stage_failed", "stage": name, "summary": summary})
        fb = str(feedback or "").strip()
        if fb and fb.lower() != "skip" and tm is not None:
            tm.content = f"{signal}\n\n[Developer guidance — follow this]: {fb}"
        return result

    return [_hitl]
