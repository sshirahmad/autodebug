"""Tests for the interactive served graph (autodebug/graph/interactive.py).

The graph is ``prepare → clone → manager`` where ``manager`` is the Manager agent
as a subgraph node. These exercise the wiring WITHOUT an LLM or Docker: the Manager
node is stubbed (build_graph(manager_node=...)) and clone_repo is monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.graph import interactive as gi  # noqa: E402
from autodebug.state import DebugState, PipelineStage  # noqa: E402


class TestHelpers:
    def test_latest_human_text_prefers_last_human_turn(self):
        msgs = [HumanMessage(content="first"), AIMessage(content="mid"),
                HumanMessage(content="the bug report")]
        assert gi._latest_human_text(msgs) == "the bug report"

    def test_ds_rebuilds_from_debug_channel(self):
        ds = DebugState(repo_url="u", bug_report="b", repo_volume="vol")
        assert gi._ds({"debug": ds.model_dump()}).repo_volume == "vol"

    def test_split_repo_and_report_extracts_url_from_message(self):
        # Agent Chat UI offers only a message box, so a run must be drivable from the
        # message alone: first URL -> repo, the rest -> bug report.
        url, rep = gi._split_repo_and_report(
            "https://github.com/psf/black\nblack crashes when /dev/shm is unavailable")
        assert url == "https://github.com/psf/black"
        assert rep == "black crashes when /dev/shm is unavailable"

    def test_split_repo_drops_bare_repo_label(self):
        url, rep = gi._split_repo_and_report("Repo: https://github.com/psf/black\nBug: it crashes")
        assert url == "https://github.com/psf/black"
        assert rep == "Bug: it crashes"

    def test_split_repo_no_url_keeps_whole_report(self):
        url, rep = gi._split_repo_and_report("no url here, just a report")
        assert url == "" and rep == "no url here, just a report"

    async def test_prepare_parses_repo_from_message_when_config_absent(self, monkeypatch):
        # No repo_url in config -> prepare must parse it out of the chat message.
        def clone(ds):
            ds.repo_volume = "v"; ds.stage = PipelineStage.REPRO; return ds
        monkeypatch.setattr(gi, "clone_repo", clone)
        seen = {}
        g = gi.build_graph(manager_node=lambda s: seen.update(debug=s["debug"]) or {"messages": []})
        await g.ainvoke(
            {"messages": [HumanMessage(content="https://github.com/psf/black\nit crashes")]},
            config={"configurable": {"thread_id": "t"}},  # note: no repo_url
        )
        assert seen["debug"]["repo_url"] == "https://github.com/psf/black"
        assert seen["debug"]["bug_report"] == "it crashes"

    def test_run_context_schema_declares_repo_url(self):
        # The declared context schema is what makes Studio render a config form.
        assert "repo_url" in gi.RunContext.__annotations__
        assert "manager" in gi.graph.get_graph().nodes

    async def test_prepare_pasted_issue_url_yields_repo_and_folds_issue(self, monkeypatch):
        # A pasted issue/PR URL doubles as the repo AND the issue, whose text is
        # fetched and folded into the bug report (fetch stubbed — no network).
        monkeypatch.setattr(gi, "fetch_issue", lambda url: "Issue title\n\nissue body")

        def clone(ds):
            ds.repo_volume = "v"; ds.stage = PipelineStage.REPRO; return ds
        monkeypatch.setattr(gi, "clone_repo", clone)

        seen = {}
        g = gi.build_graph(manager_node=lambda s: seen.update(debug=s["debug"]) or {"messages": []})
        await g.ainvoke(
            {"messages": [HumanMessage(content="https://github.com/psf/black/issues/42\nit crashes")]},
            config={"configurable": {"thread_id": "t"}},
        )
        dbg = seen["debug"]
        assert dbg["repo_url"] == "https://github.com/psf/black"          # derived from issue URL
        assert dbg["github_issue_url"] == "https://github.com/psf/black/issues/42"
        assert "it crashes" in dbg["bug_report"] and "issue body" in dbg["bug_report"]


class TestGraphWiring:
    def test_graph_exposes_shared_channels(self):
        # Studio/Agent Chat UI need `messages`; the Manager subgraph shares `debug`.
        for ch in ("messages", "debug", "fsm_phase"):
            assert ch in gi.graph.channels
        assert "manager" in gi.graph.get_graph().nodes

    def _fake_clone(self, monkeypatch, *, fail=False):
        def clone(ds: DebugState) -> DebugState:
            if fail:
                raise RuntimeError("docker down")
            ds.repo_volume = "vol-test"
            ds.stage = PipelineStage.REPRO
            return ds
        monkeypatch.setattr(gi, "clone_repo", clone)

    async def test_prepare_seeds_debug_and_manager_receives_cloned_state(self, monkeypatch):
        self._fake_clone(monkeypatch)
        seen = {}

        def stub_manager(state):
            seen["debug"] = state["debug"]
            return {"messages": [AIMessage(content="✅ manager ran")]}

        g = gi.build_graph(manager_node=stub_manager)
        cfg = {"configurable": {"thread_id": "t1", "repo_url": "https://x/y"}}
        out = await g.ainvoke({"messages": [HumanMessage(content="it crashes")]}, config=cfg)

        # prepare built debug from config + message; clone populated repo_volume.
        assert seen["debug"]["repo_url"] == "https://x/y"
        assert seen["debug"]["bug_report"] == "it crashes"
        assert seen["debug"]["repo_volume"] == "vol-test"
        assert any("manager ran" in str(getattr(m, "content", "")) for m in out["messages"])

    async def test_record_node_writes_memory_episode(self, monkeypatch):
        # Every route persists a memory episode after the Manager (best-effort).
        self._fake_clone(monkeypatch)
        calls = {}
        import autodebug.memory as mem
        monkeypatch.setattr(mem, "store_agent_run",
                            lambda stage, ds: calls.update(stage=stage, repo=ds.repo_url))

        g = gi.build_graph(manager_node=lambda s: {"messages": [AIMessage(content="done")]})
        await g.ainvoke({"messages": [HumanMessage(content="b")]},
                        config={"configurable": {"thread_id": "t", "repo_url": "https://x/y"}})
        assert calls.get("stage") == "manager" and calls.get("repo") == "https://x/y"

    async def test_clone_failure_skips_manager_and_ends_failed(self, monkeypatch):
        self._fake_clone(monkeypatch, fail=True)
        ran = {"manager": False}

        def stub_manager(state):
            ran["manager"] = True
            return {}

        g = gi.build_graph(manager_node=stub_manager)
        out = await g.ainvoke(
            {"messages": [HumanMessage(content="x")]},
            config={"configurable": {"thread_id": "t2", "repo_url": "u"}},
        )
        assert ran["manager"] is False                       # never reached the Manager
        assert DebugState(**out["debug"]).stage == PipelineStage.FAILED
