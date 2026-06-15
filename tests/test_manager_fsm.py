"""Tests for the Manager FSM, its tool-gating/prompt selection, the managed-agent
tools, and the manager config.

These deliberately avoid importing the full `autodebug.agents` /
`autodebug.tools` packages (which pull in optional heavy deps). We exercise the
FSM logic directly and stub the sub-agent drivers via sys.modules so the manager
tools can be tested without Docker or an LLM.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.fsm import (  # noqa: E402
    FSM, MANAGER_ALLOWED_TOOLS, ManagerPhase, allowed_tool_names, filter_tools,
    fsm_tool_enforce, select_prompt,
)
from autodebug.state import DebugState, ReproResult, BisectResult, RootCauseResult, FixResult  # noqa: E402


# ---------------------------------------------------------------------------
# FSM core
# ---------------------------------------------------------------------------

class TestFSM:
    def test_starts_in_init(self):
        assert FSM().phase == ManagerPhase.INIT

    def test_transition_records_history_and_is_terminal(self):
        fsm = FSM()
        fsm.to(ManagerPhase.REPRODUCED)
        fsm.to(ManagerPhase.BISECTED)
        assert fsm.phase == ManagerPhase.BISECTED
        assert fsm.history == [ManagerPhase.INIT, ManagerPhase.REPRODUCED]
        assert not fsm.is_terminal
        fsm.to(ManagerPhase.DONE)
        assert fsm.is_terminal

    def test_failed_is_terminal(self):
        fsm = FSM()
        fsm.to(ManagerPhase.FAILED)
        assert fsm.is_terminal


# ---------------------------------------------------------------------------
# Tool gating + prompt selection (the pure helpers the middleware wraps)
# ---------------------------------------------------------------------------

class _FakeTool:
    def __init__(self, name):
        self.name = name


class TestToolGate:
    def _tools(self):
        return [_FakeTool(n) for n in (
            "run_repro_agent", "run_bisect_agent", "run_root_cause_agent",
            "run_fix_agent", "finish",
        )]

    def test_init_exposes_repro_and_finish(self):
        # `finish` is always available (escape hatch); INIT otherwise only repros.
        fsm = FSM()  # INIT
        names = {t.name for t in filter_tools(fsm, self._tools(), MANAGER_ALLOWED_TOOLS)}
        assert names == {"run_repro_agent", "finish"}

    def test_analyzed_exposes_fix_root_cause_and_finish(self):
        fsm = FSM(phase=ManagerPhase.ANALYZED)
        names = {t.name for t in filter_tools(fsm, self._tools(), MANAGER_ALLOWED_TOOLS)}
        assert names == {"run_fix_agent", "run_root_cause_agent", "finish"}

    def test_finish_allowed_in_every_phase(self):
        for phase in MANAGER_ALLOWED_TOOLS:
            assert "finish" in MANAGER_ALLOWED_TOOLS[phase], f"{phase} cannot finish"

    def test_revising_opens_loopback_and_finish(self):
        fsm = FSM(phase=ManagerPhase.REVISING)
        names = {t.name for t in filter_tools(fsm, self._tools(), MANAGER_ALLOWED_TOOLS)}
        assert names == {
            "run_repro_agent", "run_root_cause_agent", "run_fix_agent", "finish",
        }

    def test_unknown_phase_leaves_tools_untouched(self):
        fsm = FSM(phase=ManagerPhase.DONE)  # not in the allowed map
        tools = self._tools()
        assert filter_tools(fsm, tools, MANAGER_ALLOWED_TOOLS) == tools

    def test_allowed_map_only_references_real_tools(self):
        real = {
            "run_repro_agent", "run_bisect_agent", "run_root_cause_agent",
            "run_fix_agent", "finish",
        }
        for names in MANAGER_ALLOWED_TOOLS.values():
            assert set(names) <= real


class TestToolEnforcement:
    """The wrap_tool_call guard blocks out-of-phase tools at EXECUTION time —
    the model-side gate only controls what's advertised, not what runs."""

    def _run(self, fsm, name):
        from types import SimpleNamespace
        from langchain_core.messages import ToolMessage

        mw = fsm_tool_enforce(fsm, MANAGER_ALLOWED_TOOLS)
        ran = {"n": 0}

        def handler(req):
            ran["n"] += 1
            return ToolMessage(content="EXECUTED", tool_call_id=req.tool_call["id"],
                               name=req.tool_call["name"])

        req = SimpleNamespace(tool_call={"name": name, "id": "x"}, tool=None,
                              state={}, runtime=None)
        return mw.wrap_tool_call(req, handler), ran["n"]

    def test_blocks_disallowed_tool_without_running_it(self):
        msg, ran = self._run(FSM(), "run_fix_agent")  # init -> fix not allowed
        assert msg.status == "error" and ran == 0
        assert "not available in phase" in msg.content

    def test_allows_permitted_tool(self):
        msg, ran = self._run(FSM(), "run_repro_agent")  # init -> repro allowed
        assert msg.content == "EXECUTED" and ran == 1

    def test_always_allows_write_todos(self):
        msg, ran = self._run(FSM(), "write_todos")
        assert msg.content == "EXECUTED" and ran == 1

    def test_permission_tracks_phase(self):
        msg, ran = self._run(FSM(phase=ManagerPhase.ANALYZED), "run_fix_agent")
        assert msg.content == "EXECUTED" and ran == 1

    def test_allowed_tool_names_matches_gate(self):
        fsm = FSM(phase=ManagerPhase.ANALYZED)
        assert allowed_tool_names(fsm, MANAGER_ALLOWED_TOOLS) == {
            "run_fix_agent", "run_root_cause_agent", "finish", "write_todos",
        }


class TestPromptSelection:
    def test_picks_prompt_for_phase(self):
        prompts = {p.value: f"prompt for {p.value}" for p in ManagerPhase}
        fsm = FSM(phase=ManagerPhase.BISECTED)
        assert select_prompt(fsm, prompts) == "prompt for bisected"

    def test_appends_progress_block(self):
        prompts = {"init": "BASE"}
        out = select_prompt(FSM(), prompts, progress_fn=lambda: "PROGRESS")
        assert out == "BASE\n\nPROGRESS"

    def test_falls_back_to_init_when_phase_missing(self):
        prompts = {"init": "fallback"}
        fsm = FSM(phase=ManagerPhase.REVISING)
        assert select_prompt(fsm, prompts) == "fallback"


# ---------------------------------------------------------------------------
# Managed-agent tools — transitions + signals (sub-agents stubbed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_agents(monkeypatch):
    """Install a stub `autodebug.agents` module whose run_* drivers are
    controllable, so the lazily-imported sub-agents in tools.manager resolve to
    them instead of triggering the real (heavy) package import."""
    mod = types.ModuleType("autodebug.agents")

    def noop(state, *, registry=None):
        return state

    for name in ("run_repro", "run_bisect", "run_root_cause", "run_fix", "run_manager"):
        setattr(mod, name, noop)
    monkeypatch.setitem(sys.modules, "autodebug.agents", mod)
    return mod


@pytest.fixture
def state():
    return DebugState(repo_url="u", bug_report="b", repo_volume="vol")


def _import_manager_tools():
    # Load the module straight from its file so we don't trigger
    # autodebug.tools.__init__ (which imports optional heavy deps). It only
    # needs autodebug.fsm (light) and lazily imports the stubbed sub-agents.
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "autodebug" / "tools" / "manager.py"
    spec = importlib.util.spec_from_file_location("_mgr_tools_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestManagerTools:
    def test_repro_confirmed_advances_to_reproduced(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM()
        fake_agents.run_repro = lambda s, *, registry=None: setattr(
            s, "repro", ReproResult(repro_script="x", error_output="boom", confirmed=True)
        )
        tool = m.make_run_repro_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert fsm.phase == ManagerPhase.REPRODUCED
        assert "REPRO CONFIRMED" in out and "boom" in out

    def test_repro_failure_stays_and_signals(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM()
        fake_agents.run_repro = lambda s, *, registry=None: None
        tool = m.make_run_repro_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert fsm.phase == ManagerPhase.INIT
        assert "REPRO FAILED" in out

    def test_bisect_requires_repro(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.REPRODUCED)
        tool = m.make_run_bisect_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})  # no repro on state
        assert "Cannot bisect" in out
        assert fsm.phase == ManagerPhase.REPRODUCED

    def test_bisect_blank_culprit_is_rejected(self, fake_agents, state):
        # A sub-agent that submits an empty SHA must NOT advance the FSM — that
        # blank culprit caused a wasted repro+bisect redo in production.
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.REPRODUCED)
        state.repro = ReproResult(repro_script="x", error_output="boom", confirmed=True)
        fake_agents.run_bisect = lambda s, *, registry=None: setattr(
            s, "bisect", BisectResult(culprit_commit="   ", commit_message="",
                                      commit_diff="", steps_taken=0)
        )
        tool = m.make_run_bisect_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert "BISECT FAILED" in out
        assert fsm.phase == ManagerPhase.REPRODUCED  # did NOT advance
        assert state.bisect is None                  # cleared for a clean retry

    def test_bisect_valid_culprit_advances(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.REPRODUCED)
        state.repro = ReproResult(repro_script="x", error_output="boom", confirmed=True)
        fake_agents.run_bisect = lambda s, *, registry=None: setattr(
            s, "bisect", BisectResult(culprit_commit="abc123", commit_message="bad commit",
                                      commit_diff="d", steps_taken=0)
        )
        tool = m.make_run_bisect_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert fsm.phase == ManagerPhase.BISECTED
        assert "CULPRIT FOUND: abc123" in out

    def test_fix_verified_advances_to_done(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.ANALYZED)
        state.root_cause = RootCauseResult(summary="s", relevant_lines=[], hypothesis="h")
        fake_agents.run_fix = lambda s, *, registry=None: setattr(
            s, "fix", FixResult(patch="p", attempts=1, test_output="ok")
        )
        tool = m.make_run_fix_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert fsm.phase == ManagerPhase.DONE
        assert "FIX VERIFIED" in out

    def test_fix_failure_enters_revising_with_options(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.ANALYZED)
        state.root_cause = RootCauseResult(summary="s", relevant_lines=[], hypothesis="h")
        fake_agents.run_fix = lambda s, *, registry=None: None  # no state.fix
        tool = m.make_run_fix_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})
        assert fsm.phase == ManagerPhase.REVISING
        assert "FIX FAILED" in out and "run_repro_agent" in out

    def test_re_repro_during_revising_stays_in_revising(self, fake_agents, state):
        m = _import_manager_tools()
        fsm = FSM(phase=ManagerPhase.REVISING)
        fake_agents.run_repro = lambda s, *, registry=None: setattr(
            s, "repro", ReproResult(repro_script="x", error_output="e", confirmed=True)
        )
        tool = m.make_run_repro_agent_tool(state=state, registry=None, fsm=fsm)
        tool.invoke({})
        assert fsm.phase == ManagerPhase.REVISING

    def test_subagent_crash_returns_signal_not_exception(self, fake_agents, state):
        """A sub-agent raising (e.g. a transient provider error) must not abort the
        manager — the tool catches it and returns a retryable failure signal."""
        m = _import_manager_tools()
        fsm = FSM()

        def boom(s, *, registry=None):
            raise RuntimeError("provider returned error 400")

        fake_agents.run_repro = boom
        tool = m.make_run_repro_agent_tool(state=state, registry=None, fsm=fsm)
        out = tool.invoke({})  # must not raise
        assert "crashed" in out.lower() and "provider returned error 400" in out
        assert fsm.phase == ManagerPhase.INIT  # phase unchanged — manager can retry

    def test_finish_success_and_failure(self, fake_agents, state):
        m = _import_manager_tools()

        def _finish(fsm, outcome, summary):
            tool = m.make_finish_tool(state=state, fsm=fsm)
            res = tool.invoke({"name": "finish",
                               "args": {"outcome": outcome, "summary": summary},
                               "id": "c1", "type": "tool_call"})
            return res.update["outcome"]   # the Command's outcome-channel write

        fsm = FSM(phase=ManagerPhase.REVISING)
        assert _finish(fsm, "success", "done") == {"outcome": "success", "summary": "done"}
        assert fsm.phase == ManagerPhase.DONE

        fsm2 = FSM(phase=ManagerPhase.REVISING)
        assert _finish(fsm2, "failed", "stuck")["outcome"] == "failed"
        assert fsm2.phase == ManagerPhase.FAILED


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

class TestManagerConfig:
    def test_prompt_map_has_a_prompt_per_phase(self):
        from autodebug.config.loader import ConfigLoader
        prompts = ConfigLoader().resolve_prompt_map("config/prompts/manager.yaml")
        for phase in (ManagerPhase.INIT, ManagerPhase.REPRODUCED, ManagerPhase.BISECTED,
                      ManagerPhase.ANALYZED, ManagerPhase.REVISING):
            assert prompts.get(phase.value), f"missing prompt for {phase.value}"

    def test_manager_json_is_valid_agent_config(self):
        import json
        from autodebug.config.schema import AgentConfig
        data = json.loads(Path("config/agents/manager.json").read_text())
        known = {k: v for k, v in data.items() if k in AgentConfig.model_fields}
        cfg = AgentConfig(**known)
        assert set(cfg.tools) == {
            "run_repro_agent", "run_bisect_agent", "run_root_cause_agent",
            "run_fix_agent", "finish",
        }
        # tool-call limits must reference real manager tools
        assert set(cfg.tool_call_limits) <= set(cfg.tools)
