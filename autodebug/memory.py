"""Persistent memory store for AutoDebug agents.

Uses LangMem + SqliteStore with vector search so agents can retrieve
experiences from past debugging sessions before starting a new one.

Env vars:
    AUTODEBUG_MEMORY_MODEL   — LLM used to extract memories
                               (default: openrouter/owl-alpha via OpenRouter)
    AUTODEBUG_EMBED_MODEL    — embedding model string
                               (default: huggingface:sentence-transformers/all-MiniLM-L6-v2, runs locally)
    AUTODEBUG_EMBED_DIMS     — embedding dimensions (default: 384)
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.store.base import IndexConfig
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager, create_search_memory_tool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

_DB_PATH      = Path(__file__).resolve().parents[1] / "data" / "memory.db"
_EMBED_MODEL  = os.getenv("AUTODEBUG_EMBED_MODEL", "huggingface:sentence-transformers/all-MiniLM-L6-v2")
_EMBED_DIMS   = int(os.getenv("AUTODEBUG_EMBED_DIMS", "384"))
_MEMORY_MODEL = os.getenv("AUTODEBUG_MEMORY_MODEL", "openrouter/owl-alpha")
# Provider for the memory model. None => build_model falls back to the agent's
# provider (AUTODEBUG_MODEL_PROVIDER) — set this when the memory model needs a
# different provider/route than the agent's (e.g. agent on Anthropic, memory on
# an OpenRouter model).
_MEMORY_PROVIDER = os.getenv("AUTODEBUG_MEMORY_PROVIDER") or None

_EXTRACT_INSTRUCTIONS = (
    "Extract debugging actions, observations, and results from this debugging session. "
    "Focus on: what scripts or patches worked, which files were relevant, what error patterns "
    "appeared, and any strategies that succeeded or failed. Record information that would help "
    "a future agent debug a similar bug in the same or related project."
)


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------

class DebugAction(BaseModel):
    """An action taken during a debugging session."""
    action_type: str = Field(description="Tool used: read_file, run_script, apply_patch, etc.")
    target: str = Field(description="File path, script content, or other target of the action")
    outcome: str = Field(description="What the action returned or produced")


class DebugObservation(BaseModel):
    """An important observation about the bug, codebase, or debugging strategy."""
    project: str = Field(description="Project name (e.g. ansible, pandas, black)")
    observation: str = Field(description="What was observed about the bug or codebase")
    context: str = Field(description="What led to this observation")


class DebugResult(BaseModel):
    """A final result produced by an agent stage."""
    stage: str = Field(description="Agent stage: repro, bisect, root_cause, or fix")
    project: str = Field(description="Project name")
    success: bool = Field(description="Whether the stage succeeded")
    summary: str = Field(description="Summary of what was found or produced")


# ---------------------------------------------------------------------------
# Store (singleton)
# ---------------------------------------------------------------------------

_store: SqliteStore | None = None


def memory_store() -> SqliteStore:
    global _store
    if _store is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, isolation_level=None)
        # memory.db is a single store shared across runs (cross-bug learning is the
        # point). Make it safe for concurrent worker processes the same way as the
        # checkpointer: WAL lets a reader and a writer proceed at once, and the busy
        # timeout makes a contending writer WAIT instead of failing "database is
        # locked". (Failures here are swallowed, so without this, memory writes are
        # just silently dropped under --workers contention.)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(os.getenv('AUTODEBUG_SQLITE_BUSY_MS', '30000'))}")
        store = SqliteStore(
            conn,
            index=IndexConfig(
                dims=_EMBED_DIMS,
                embed=_EMBED_MODEL,
                fields=["observation", "summary", "outcome", "context"],
            ),
        )
        store.setup()
        _store = store
    return _store


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------

def _get_manager(namespace: tuple[str, ...]):
    from autodebug.agents.base import build_model
    model = build_model(model_id=_MEMORY_MODEL, provider=_MEMORY_PROVIDER)
    return create_memory_store_manager(
        model,
        schemas=[DebugAction, DebugObservation, DebugResult],
        instructions=_EXTRACT_INSTRUCTIONS,
        namespace=namespace,
        store=memory_store(),
        enable_inserts=True,
        enable_deletes=False,
    )


def store_agent_run(stage: str, state) -> None:
    """Extract and persist memories from a completed agent stage. Never raises."""
    try:
        project = _project_from_url(state.repo_url)
        messages = [HumanMessage(content=_build_summary(stage, project, state))]
        manager = _get_manager(("autodebug", stage))
        manager.invoke({"messages": messages}, config={"recursion_limit": 500})
    except Exception as exc:
        # Memory failures must never break the pipeline — but surface them.
        logger.warning("Memory store failed for stage %r: %s: %s",
                       stage, type(exc).__name__, exc)


def _project_from_url(repo_url: str) -> str:
    match = re.search(r"github\.com/[^/]+/([^/]+?)(?:\.git)?$", repo_url or "")
    return match.group(1) if match else "unknown"


def _build_summary(stage: str, project: str, state) -> str:
    lines = [
        f"Stage: {stage}",
        f"Project: {project}",
        f"Bug report: {state.bug_report}",
    ]
    if state.repro:
        lines += [
            f"Repro script:\n{state.repro.repro_script}",
            f"Error output: {state.repro.error_output}",
        ]
    if state.bisect:
        lines += [
            f"Culprit commit: {state.bisect.culprit_commit}",
            f"Commit message: {state.bisect.commit_message}",
            f"Diff (truncated):\n{state.bisect.commit_diff[:1000]}",
        ]
    if state.root_cause:
        lines += [
            f"Root cause hypothesis: {state.root_cause.hypothesis}",
            f"Summary: {state.root_cause.summary}",
            f"Relevant lines: {', '.join(state.root_cause.relevant_lines)}",
        ]
    if state.fix:
        lines += [f"Fix patch:\n{state.fix.patch[:500]}"]
    if state.error:
        lines += [f"Failure reason: {state.error}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search tool factory (auto-discovered via make_*_tool naming convention)
# ---------------------------------------------------------------------------

_SEARCH_INSTRUCTIONS_BY_AGENT = {
    "repro": (
        "Search your past reproduction attempts for similar bugs. Look for scripts that "
        "successfully reproduced comparable errors, useful import patterns, or environment "
        "setups that worked. Avoid repeating approaches that previously failed."
    ),
    "bisect": (
        "Search your past bisect runs for similar projects or error patterns. Look for "
        "strategies that efficiently narrowed down culprit commits, and ranges that were "
        "previously identified as good starting points."
    ),
    "root_cause": (
        "Search your past root cause analyses for similar bugs or projects. Look for "
        "relevant file paths, hypothesis patterns, or analysis strategies that worked."
    ),
    "fix": (
        "Search your past fix attempts for similar bugs. Look for patch patterns that "
        "worked, approaches that failed, or test commands that reliably validated fixes."
    ),
}


def make_search_memory_tool(agent_name: str = "unknown", **_):
    """Factory: returns a LangChain tool that searches this agent's past memories only.

    The underlying LangMem tool exposes a rich schema (query, namespace filters,
    limits, etc.) that models often hallucinate field names for — we've seen
    `{'key': '<uuid>'}` instead of `{'query': '<text>'}`. Wrap it with a clean
    single-arg `query: str` signature so the LLM can only get it right.
    """
    namespace = ("autodebug", agent_name)
    instructions = _SEARCH_INSTRUCTIONS_BY_AGENT.get(
        agent_name,
        "Search past debugging sessions for relevant patterns and strategies.",
    )
    underlying = create_search_memory_tool(
        namespace=namespace,
        store=memory_store(),
        instructions=instructions,
    )

    @tool(parse_docstring=True)
    def search_memory(query: str) -> str:
        """Search past debugging sessions for relevant patterns.

        Args:
            query: Natural-language description of what you're looking for
                (e.g. "ansible collection MANIFEST.json verify error").
        """
        try:
            return str(underlying.invoke({"query": query}))
        except Exception as exc:
            logger.warning("Memory search failed: %s: %s", type(exc).__name__, exc)
            return f"memory search failed: {exc}"

    return search_memory
