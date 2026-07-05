"""Per-agent graph-state schemas.

Each agent's submission lives in a dedicated state channel (not an external
list), so LangGraph's checkpointer persists it automatically — that is what makes
a budget-killed run resumable: re-read the channel from the saved thread instead
of re-deriving the result. Artifacts are stored as plain dicts (``model_dump()``)
to avoid serializing custom classes through the checkpointer; the drivers rebuild
the pydantic models on the way out.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Optional

from langchain.agents import AgentState


class ReproAgentState(AgentState):
    repro: Optional[dict]


class BisectAgentState(AgentState):
    bisect: Optional[dict]


class RootCauseAgentState(AgentState):
    root_cause: Optional[dict]


class FixAgentState(AgentState):
    # `add` reducer: each apply_patch appends its record to the running list.
    patches: Annotated[list, add]
    fix: Optional[dict]


class ManagerAgentState(AgentState):
    outcome: Optional[dict]
    # The whole pipeline state (DebugState.model_dump()) and the FSM phase live in
    # the graph state so they're checkpointed per-thread. This lets the Manager run
    # as a SERVED, resumable subgraph — its sub-agent results and phase survive an
    # interrupt instead of living in external closures (see autodebug/tools/manager.py
    # and autodebug/agents/manager.py). Plain dict/str to keep the checkpoint
    # serializer happy; tools rebuild the pydantic DebugState on the way in/out.
    debug: Optional[dict]
    fsm_phase: Optional[str]
    # Extra session-budget granted by a human at a budget-HITL interrupt (each resume
    # adds another window so the Manager can keep going). 0/None until the first grant.
    budget_extra: Optional[float]
