"""Persistent, bug-keyed resume for the agents, built on LangGraph's checkpointer.

Each sub-agent already runs on a checkpointer; we make it a persistent
``SqliteSaver`` (``.autodebug/agents.sqlite``) keyed by the bug. A run that dies
mid-pipeline (e.g. the session budget is exhausted at the fix stage) can then
recover the already-paid-for repro / bisect / root_cause WITHOUT re-spending
tokens, leaving the fresh session's whole budget for the fix. The fix itself is
never cached — it is exactly the stage we want to (re)attempt.

The checkpointer persists each attempt's message history. We reconstruct the
structured artifact from the *accepted* ``submit_*`` tool-call, recomputing the
parts that aren't pure args so the restored result stays faithful to the fresh
checkout: the repro re-runs its script (and is dropped if it no longer fails),
bisect re-reads the commit. root_cause is pure reasoning, rebuilt from its args.

Toggles:
  AUTODEBUG_RESUME        "0" disables the read-side short-circuit (states are
                          still saved). Default on.
  AUTODEBUG_CHECKPOINT_DIR  directory holding agents.sqlite (default .autodebug).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path

from autodebug.state import BisectResult, ReproResult, RootCauseResult

logger = logging.getLogger(__name__)

_saver = None


def resume_enabled() -> bool:
    return os.getenv("AUTODEBUG_RESUME", "1") != "0"


def get_saver():
    """Shared persistent SqliteSaver (one connection per process; agents run
    sequentially). Falls back to an in-memory saver if sqlite can't be opened, so
    checkpointing never blocks a run."""
    global _saver
    if _saver is not None:
        return _saver
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(os.getenv("AUTODEBUG_CHECKPOINT_DIR", ".autodebug")) / "agents.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        _saver = saver
    except Exception as exc:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning(
            "Persistent checkpoint store unavailable (%s: %s); resume is disabled "
            "this run — using an in-memory saver.", type(exc).__name__, exc,
        )
        _saver = InMemorySaver()
    return _saver


def bug_key(state) -> str:
    """Stable id for a (repo, checkout, bug) so a different bug never resumes from
    a stale checkpoint."""
    raw = f"{state.repo_url}\x00{state.ref or ''}\x00{state.bug_report}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def thread_id(agent: str, key: str, attempt: int) -> str:
    return f"{agent}-{key}-{attempt}"


def clear(agent: str, key: str, max_attempts: int) -> None:
    """Drop a bug's persisted attempt threads for an agent before a fresh live run
    so we never append onto stale state. Best-effort."""
    saver = get_saver()
    for attempt in range(max_attempts + 1):
        try:
            saver.delete_thread(thread_id(agent, key, attempt))
        except Exception:
            pass


def read_live(agent, invoke_config, channel: str):
    """Read a result channel from the just-run agent's persisted state — works
    after a normal finish AND after a crash (the checkpointer holds whatever the
    submit tool wrote before the budget tripped)."""
    try:
        return (agent.get_state(invoke_config).values or {}).get(channel)
    except Exception as exc:
        logger.warning("Could not read result channel %r from agent state: %s: %s",
                       channel, type(exc).__name__, exc)
        return None


def clear_one(agent: str, key: str, attempt: int) -> None:
    """Drop a single attempt's thread (e.g. a driver-rejected submission) so it
    can't be resumed later."""
    try:
        get_saver().delete_thread(thread_id(agent, key, attempt))
    except Exception:
        pass


def _channel(agent: str, key: str, max_attempts: int, channel: str):
    """The result dict a prior run wrote to `channel`, newest attempt first.

    A rejected submit never writes its channel, so anything found here was an
    accepted submission — no message parsing or markers needed.
    """
    if not resume_enabled():
        return None
    saver = get_saver()
    for attempt in range(max_attempts, -1, -1):
        try:
            ck = saver.get({"configurable": {"thread_id": thread_id(agent, key, attempt)}})
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s: %s",
                           thread_id(agent, key, attempt), type(exc).__name__, exc)
            continue
        val = (ck or {}).get("channel_values", {}).get(channel)
        if val:
            return val
    return None


def _thread_messages(tid: str) -> list:
    try:
        ck = get_saver().get({"configurable": {"thread_id": tid}})
    except Exception as exc:
        logger.warning("Could not read checkpoint %s: %s: %s", tid, type(exc).__name__, exc)
        return []
    return (ck or {}).get("channel_values", {}).get("messages", []) or []


def last_attempt_artifact(agent: str, key: str, max_attempts: int, tool_name: str):
    """Args of the most recent `tool_name` call across this bug's attempt threads
    (newest first) — i.e. the candidate a prior *failed* attempt actually tried
    (a repro script, a culprit SHA). Lets the next attempt refine instead of
    restarting blind. Returns the args dict, or None.

    Reads the message history (failed submits write no channel), so call it BEFORE
    `clear()` wipes the prior threads.
    """
    if not resume_enabled():
        return None
    for attempt in range(max_attempts, -1, -1):
        last = None
        for m in _thread_messages(thread_id(agent, key, attempt)):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("name") == tool_name:
                    last = tc.get("args") or {}
        if last is not None:
            return last
    return None


def cached_repro(key: str, max_attempts: int, sandbox) -> ReproResult | None:
    data = _channel("repro", key, max_attempts, "repro")
    if not data or not data.get("repro_script"):
        return None
    # Re-validate against the fresh checkout: a cached script that no longer fails
    # is stale and must not be trusted as the oracle.
    run = sandbox.run_script(data["repro_script"])
    if run.success:
        return None
    return ReproResult(
        repro_script=data["repro_script"], error_output=run.output[-4000:], confirmed=True
    )


def cached_bisect(key: str, max_attempts: int, sandbox) -> BisectResult | None:
    data = _channel("bisect", key, max_attempts, "bisect")
    if not data:
        return None
    sha = (data.get("culprit_commit") or "").strip()
    if not sha:
        return None
    from autodebug.tools import git_utils as _gu

    info = _gu.get_commit_info(sandbox, sha)  # refresh the diff against the fresh checkout
    if not info.sha:
        return None
    return BisectResult(
        culprit_commit=info.sha, commit_message=info.message,
        commit_diff=info.diff, steps_taken=0,
    )


def cached_root_cause(key: str, max_attempts: int) -> RootCauseResult | None:
    data = _channel("root_cause", key, max_attempts, "root_cause")
    if not data or not (data.get("hypothesis") or "").strip():
        return None
    return RootCauseResult(**data)
