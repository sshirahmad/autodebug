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

        mw = fsm_tool_enforce(MANAGER_ALLOWED_TOOLS)
        ran = {"n": 0}

        def handler(req):
            ran["n"] += 1
            return ToolMessage(content="EXECUTED", tool_call_id=req.tool_call["id"],
                               name=req.tool_call["name"])

        # phase now lives in the graph state, not a closure FSM.
        req = SimpleNamespace(tool_call={"name": name, "id": "x"}, tool=None,
                              state={"fsm_phase": fsm.phase.value}, runtime=None)
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


class TestAsyncMiddleware:
    """The FSM wrap_* middlewares must implement the ASYNC hooks too: Studio / Agent
    Chat UI drive the graph with ainvoke/astream, and langchain does NOT auto-derive
    awrap_model_call/awrap_tool_call from the sync version — it raises
    NotImplementedError by default (this took down Studio: 'Asynchronous implementation
    of awrap_model_call is not available')."""

    def test_all_fsm_wrap_middlewares_override_async_hooks(self):
        from langchain.agents.middleware import AgentMiddleware
        from autodebug.fsm import fsm_tool_gate, fsm_tool_enforce, stage_hitl_middleware

        gate = fsm_tool_gate(MANAGER_ALLOWED_TOOLS)
        enforce = fsm_tool_enforce(MANAGER_ALLOWED_TOOLS)
        hitl = stage_hitl_middleware()[0]
        assert type(gate).awrap_model_call is not AgentMiddleware.awrap_model_call
        assert type(enforce).awrap_tool_call is not AgentMiddleware.awrap_tool_call
        assert type(hitl).awrap_tool_call is not AgentMiddleware.awrap_tool_call

    async def test_gate_awrap_model_call_filters_and_delegates(self):
        # The async hook mirrors the sync one: filter tools for the phase, then await.
        from types import SimpleNamespace

        from autodebug.fsm import fsm_tool_gate
        mw = fsm_tool_gate(MANAGER_ALLOWED_TOOLS)
        req = SimpleNamespace(state={"fsm_phase": ManagerPhase.INIT.value},
                              tools=[_FakeTool(n) for n in ("run_repro_agent", "run_fix_agent", "finish")])

        async def handler(r):
            return {"tools_seen": {t.name for t in r.tools}}

        out = await mw.awrap_model_call(req, handler)
        assert out["tools_seen"] == {"run_repro_agent", "finish"}  # INIT gate applied


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
    """The pure `_*_step` functions hold the transition logic: they mutate the
    DebugState via the (stubbed) driver and return (new_phase | None, signal). The
    `@tool` wrappers are thin InjectedState adapters, exercised separately."""

    def test_repro_confirmed_advances_to_reproduced(self, fake_agents, state):
        m = _import_manager_tools()
        fake_agents.run_repro = lambda s, *, registry=None: setattr(
            s, "repro", ReproResult(repro_script="x", error_output="boom", confirmed=True)
        )
        new_phase, signal = m._repro_step(state, ManagerPhase.INIT.value, None)
        assert new_phase == ManagerPhase.REPRODUCED
        assert "REPRO CONFIRMED" in signal and "boom" in signal

    def test_repro_failure_stays_and_signals(self, fake_agents, state):
        m = _import_manager_tools()
        fake_agents.run_repro = lambda s, *, registry=None: None
        new_phase, signal = m._repro_step(state, ManagerPhase.INIT.value, None)
        assert new_phase is None  # stays in current phase
        assert "REPRO FAILED" in signal

    def test_bisect_requires_repro(self, fake_agents, state):
        m = _import_manager_tools()
        new_phase, signal = m._bisect_step(state, ManagerPhase.REPRODUCED.value, None)
        assert "Cannot bisect" in signal and new_phase is None

    def test_bisect_blank_culprit_is_rejected(self, fake_agents, state):
        # A sub-agent that submits an empty SHA must NOT be treated as a real find:
        # the culprit is cleared and the FSM stays (no advance on a bogus culprit).
        m = _import_manager_tools()
        state.repro = ReproResult(repro_script="x", error_output="boom", confirmed=True)
        fake_agents.run_bisect = lambda s, *, registry=None: setattr(
            s, "bisect", BisectResult(culprit_commit="   ", commit_message="",
                                      commit_diff="", steps_taken=0)
        )
        new_phase, signal = m._bisect_step(state, ManagerPhase.REPRODUCED.value, None)
        assert "INCONCLUSIVE" in signal and "run_root_cause_agent" in signal
        assert new_phase is None        # did NOT advance on a blank culprit
        assert state.bisect is None     # cleared

    def test_bisect_valid_culprit_advances(self, fake_agents, state):
        m = _import_manager_tools()
        state.repro = ReproResult(repro_script="x", error_output="boom", confirmed=True)
        fake_agents.run_bisect = lambda s, *, registry=None: setattr(
            s, "bisect", BisectResult(culprit_commit="abc123", commit_message="bad commit",
                                      commit_diff="d", steps_taken=0)
        )
        new_phase, signal = m._bisect_step(state, ManagerPhase.REPRODUCED.value, None)
        assert new_phase == ManagerPhase.BISECTED
        assert "CULPRIT FOUND: abc123" in signal

    def test_fix_verified_advances_to_done(self, fake_agents, state):
        m = _import_manager_tools()
        state.root_cause = RootCauseResult(summary="s", relevant_lines=[], hypothesis="h")
        fake_agents.run_fix = lambda s, *, registry=None: setattr(
            s, "fix", FixResult(patch="p", attempts=1, test_output="ok")
        )
        new_phase, signal = m._fix_step(state, ManagerPhase.ANALYZED.value, None)
        assert new_phase == ManagerPhase.DONE
        assert "FIX VERIFIED" in signal

    def test_fix_failure_enters_revising_with_options(self, fake_agents, state):
        m = _import_manager_tools()
        state.root_cause = RootCauseResult(summary="s", relevant_lines=[], hypothesis="h")
        fake_agents.run_fix = lambda s, *, registry=None: None  # no state.fix
        new_phase, signal = m._fix_step(state, ManagerPhase.ANALYZED.value, None)
        assert new_phase == ManagerPhase.REVISING
        assert "FIX FAILED" in signal and "run_repro_agent" in signal

    def test_re_repro_during_revising_stays_in_revising(self, fake_agents, state):
        m = _import_manager_tools()
        fake_agents.run_repro = lambda s, *, registry=None: setattr(
            s, "repro", ReproResult(repro_script="x", error_output="e", confirmed=True)
        )
        new_phase, _ = m._repro_step(state, ManagerPhase.REVISING.value, None)
        assert new_phase is None  # stays in REVISING

    def test_subagent_crash_returns_signal_not_exception(self, fake_agents, state):
        """A sub-agent raising (e.g. a transient provider error) must not abort the
        manager — the step catches it and returns a retryable failure signal."""
        m = _import_manager_tools()

        def boom(s, *, registry=None):
            raise RuntimeError("provider returned error 400")

        fake_agents.run_repro = boom
        new_phase, signal = m._repro_step(state, ManagerPhase.INIT.value, None)  # must not raise
        assert "crashed" in signal.lower() and "provider returned error 400" in signal
        assert new_phase is None  # phase unchanged — manager can retry

    def test_tool_wrapper_injects_state_and_writes_debug(self, fake_agents, state):
        # The @tool wrapper reads `debug`/`fsm_phase` from injected state and returns
        # a Command that writes the updated debug + phase + signal back.
        m = _import_manager_tools()
        fake_agents.run_repro = lambda s, *, registry=None: setattr(
            s, "repro", ReproResult(repro_script="x", error_output="boom", confirmed=True)
        )
        tool = m.make_run_repro_agent_tool(registry=None)
        cmd = tool.invoke({"name": "run_repro_agent", "id": "c1", "type": "tool_call",
                           "args": {"state": {"debug": state.model_dump(),
                                              "fsm_phase": ManagerPhase.INIT.value}}})
        assert cmd.update["fsm_phase"] == ManagerPhase.REPRODUCED.value
        assert cmd.update["debug"]["repro"]["confirmed"] is True
        assert "REPRO CONFIRMED" in cmd.update["messages"][0].content

    def test_finish_success_and_failure(self, fake_agents, state):
        m = _import_manager_tools()

        def _finish(outcome, summary):
            tool = m.make_finish_tool(registry=None)
            res = tool.invoke({"name": "finish",
                               "args": {"outcome": outcome, "summary": summary},
                               "id": "c1", "type": "tool_call"})
            return res.update

        upd = _finish("success", "done")
        assert upd["outcome"] == {"outcome": "success", "summary": "done"}
        assert upd["fsm_phase"] == ManagerPhase.DONE.value

        upd2 = _finish("failed", "stuck")
        assert upd2["outcome"]["outcome"] == "failed"
        assert upd2["fsm_phase"] == ManagerPhase.FAILED.value


class TestStageHITL:
    """stage_hitl_middleware: pause on a blocking stage's failure, fold the human's
    reply into the tool result. interrupt() is monkeypatched to simulate a resume."""

    def _run(self, monkeypatch, tool_name, signal, resume="focus on parser.py"):
        import langgraph.types as lt
        from types import SimpleNamespace
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        monkeypatch.setattr(lt, "interrupt", lambda payload: resume)
        from autodebug.fsm import stage_hitl_middleware
        mw = stage_hitl_middleware()[0]

        tm = ToolMessage(content=signal, tool_call_id="c1")

        def handler(req):
            return Command(update={"messages": [tm], "debug": req.state["debug"]})

        st = DebugState(repo_url="u", bug_report="b",
                        repro=ReproResult(repro_script="x", error_output="e", confirmed=True))
        req = SimpleNamespace(tool_call={"name": tool_name, "id": "c1"}, tool=None,
                              state={"debug": st.model_dump()}, runtime=None)
        return mw.wrap_tool_call(req, handler)

    def test_interrupts_on_fix_failure_and_folds_feedback(self, monkeypatch):
        out = self._run(monkeypatch, "run_fix_agent", "FIX FAILED — the patch did not pass.")
        content = out.update["messages"][0].content
        assert "Developer guidance" in content and "focus on parser.py" in content

    def test_skip_leaves_signal_unchanged(self, monkeypatch):
        out = self._run(monkeypatch, "run_fix_agent", "FIX FAILED — nope.", resume="skip")
        assert "Developer guidance" not in out.update["messages"][0].content

    def test_success_signal_passes_through(self, monkeypatch):
        out = self._run(monkeypatch, "run_fix_agent", "FIX VERIFIED — all good.")
        assert out.update["messages"][0].content == "FIX VERIFIED — all good."

    def test_non_blocking_tool_ignored(self, monkeypatch):
        # bisect is optional — its INCONCLUSIVE is not a HITL trigger.
        out = self._run(monkeypatch, "run_bisect_agent", "BISECT INCONCLUSIVE — no culprit.")
        assert "Developer guidance" not in out.update["messages"][0].content


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
