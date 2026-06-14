"""Tests for persistent, bug-keyed resume (autodebug/resume.py).

Resume reads each agent's result *channel* from a SqliteSaver-backed thread —
no message parsing. These seed a channel exactly as the submit tools would (via a
graph compiled with the agent's state schema) and check reconstruction, with no
LLM or Docker.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.graph import START, END, StateGraph  # noqa: E402

from autodebug import resume  # noqa: E402
from autodebug.agent_state import (  # noqa: E402
    BisectAgentState, ReproAgentState, RootCauseAgentState,
)
from autodebug.sandbox import RunResult  # noqa: E402


def _seed(tmp_path, schema, thread: str, update: dict):
    """Compile a one-node graph with `schema`, write `update` into its channels,
    and point the resume module's shared saver at the resulting db."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(tmp_path / "agents.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    g = StateGraph(schema)
    g.add_node("n", lambda _s: update)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=saver)
    app.invoke({"messages": [HumanMessage(content="go")]}, {"configurable": {"thread_id": thread}})

    resume._saver = saver
    return saver


@pytest.fixture(autouse=True)
def _reset_saver(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTODEBUG_RESUME", "1")
    resume._saver = None
    yield
    resume._saver = None


# ---------------------------------------------------------------------------
# root_cause — pure-dict reconstruction from the channel
# ---------------------------------------------------------------------------

def test_cached_root_cause_reconstructs_from_channel(tmp_path):
    key = "K"
    rc = {"summary": "s", "relevant_lines": ["black.py:10"], "hypothesis": "h",
          "evidence": "observed boom", "fix_plan": "wrap in try"}
    _seed(tmp_path, RootCauseAgentState, resume.thread_id("root_cause", key, 0),
          {"root_cause": rc})
    got = resume.cached_root_cause(key, max_attempts=2)
    assert got is not None
    assert got.hypothesis == "h" and got.fix_plan == "wrap in try"
    assert got.relevant_lines == ["black.py:10"]


def test_cached_root_cause_none_when_channel_unset(tmp_path):
    # Thread exists (e.g. the agent ran out of budget) but never wrote the channel.
    _seed(tmp_path, RootCauseAgentState, resume.thread_id("root_cause", "K", 0), {})
    assert resume.cached_root_cause("K", max_attempts=2) is None


def test_resume_disabled_returns_none(tmp_path, monkeypatch):
    key = "K"
    _seed(tmp_path, RootCauseAgentState, resume.thread_id("root_cause", key, 0),
          {"root_cause": {"summary": "s", "relevant_lines": [], "hypothesis": "h",
                          "evidence": "", "fix_plan": ""}})
    monkeypatch.setenv("AUTODEBUG_RESUME", "0")
    assert resume.cached_root_cause(key, max_attempts=2) is None


def test_cached_root_cause_scans_newest_attempt_first(tmp_path):
    key = "K"
    saver = _seed(tmp_path, RootCauseAgentState, resume.thread_id("root_cause", key, 0),
                  {"root_cause": {"summary": "old", "relevant_lines": [], "hypothesis": "old-h",
                                  "evidence": "", "fix_plan": ""}})
    # A later attempt on the same saver/db with a different result.
    g = StateGraph(RootCauseAgentState)
    g.add_node("n", lambda _s: {"root_cause": {"summary": "new", "relevant_lines": [],
               "hypothesis": "new-h", "evidence": "", "fix_plan": ""}})
    g.add_edge(START, "n"); g.add_edge("n", END)
    g.compile(checkpointer=saver).invoke(
        {"messages": [HumanMessage(content="go")]},
        {"configurable": {"thread_id": resume.thread_id("root_cause", key, 1)}})
    assert resume.cached_root_cause(key, max_attempts=2).hypothesis == "new-h"


# ---------------------------------------------------------------------------
# repro — re-validates the cached script against the fresh checkout
# ---------------------------------------------------------------------------

def test_cached_repro_revalidates_and_accepts_failing_script(tmp_path):
    key = "K"
    _seed(tmp_path, ReproAgentState, resume.thread_id("repro", key, 0),
          {"repro": {"repro_script": "raise SystemExit(1)", "error_output": "x", "confirmed": True}})
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=1, stdout="", stderr="boom")
    rr = resume.cached_repro(key, max_attempts=2, sandbox=sandbox)
    assert rr is not None and rr.confirmed and rr.repro_script == "raise SystemExit(1)"


def test_cached_repro_drops_stale_script_that_no_longer_fails(tmp_path):
    key = "K"
    _seed(tmp_path, ReproAgentState, resume.thread_id("repro", key, 0),
          {"repro": {"repro_script": "print('ok')", "error_output": "x", "confirmed": True}})
    sandbox = MagicMock()
    sandbox.run_script.return_value = RunResult(exit_code=0, stdout="ok", stderr="")
    assert resume.cached_repro(key, max_attempts=2, sandbox=sandbox) is None


# ---------------------------------------------------------------------------
# bisect — refreshes the commit from the cached SHA
# ---------------------------------------------------------------------------

def test_cached_bisect_refreshes_commit(tmp_path):
    from autodebug.tools.git_utils import CommitInfo

    key = "K"
    _seed(tmp_path, BisectAgentState, resume.thread_id("bisect", key, 0),
          {"bisect": {"culprit_commit": "abc123", "commit_message": "old msg",
                      "commit_diff": "old", "steps_taken": 0}})
    sandbox = MagicMock()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("autodebug.tools.git_utils.get_commit_info",
                   lambda sb, sha: CommitInfo(sha="abc123full", short_sha="abc123f",
                                              message="fresh msg", author="a", date="d", diff="fresh"))
        bisect = resume.cached_bisect(key, max_attempts=2, sandbox=sandbox)
    assert bisect.culprit_commit == "abc123full" and bisect.commit_diff == "fresh"


# ---------------------------------------------------------------------------
# bug_key stability
# ---------------------------------------------------------------------------

def test_bug_key_stable_and_distinct():
    from autodebug.state import DebugState
    a = DebugState(repo_url="u", bug_report="b", ref="r")
    a2 = DebugState(repo_url="u", bug_report="b", ref="r")
    b = DebugState(repo_url="u", bug_report="DIFFERENT", ref="r")
    assert resume.bug_key(a) == resume.bug_key(a2)
    assert resume.bug_key(a) != resume.bug_key(b)
