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
MANAGER_ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    ManagerPhase.INIT.value:       ("run_repro_agent",),
    ManagerPhase.REPRODUCED.value: ("run_bisect_agent", "run_repro_agent"),
    ManagerPhase.BISECTED.value:   ("run_root_cause_agent", "run_repro_agent"),
    ManagerPhase.ANALYZED.value:   ("run_fix_agent", "run_root_cause_agent"),
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


def fsm_prompt(fsm: FSM, prompts: dict, progress_fn=None):
    """`dynamic_prompt` middleware: pick the system prompt for the current phase.

    `prompts` maps phase value (str) -> system prompt. If `progress_fn` is
    given, its return value is appended so the manager always sees a compact,
    up-to-date status block without re-deriving it.
    """

    @dynamic_prompt
    def _prompt(request) -> str:
        return select_prompt(fsm, prompts, progress_fn)

    return _prompt


def fsm_tool_gate(fsm: FSM, allowed: dict):
    """`wrap_model_call` middleware: expose only the tools allowed in this phase.

    The tools stay registered with the agent — this only controls what the model
    is offered on a given turn, which is how we steer the FSM.
    """

    @wrap_model_call
    def _gate(request, handler):
        request.tools = filter_tools(fsm, request.tools, allowed)
        return handler(request)

    return _gate


def fsm_tool_enforce(fsm: FSM, allowed: dict):
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
