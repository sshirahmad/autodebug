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
