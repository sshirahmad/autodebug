"""Tests for auto-discovery of tools and agent config registration."""

from autodebug.agents import run_bisect, run_fix, run_manager, run_repro, run_root_cause
from autodebug.registry import AutoDebugRegistry, _TOOL_FACTORIES
from autodebug.tools import _FACTORIES


EXPECTED_TOOLS = {
    "read_file", "list_files", "shell",
    "run_script", "submit_repro",
    "mark_bad", "mark_good", "mark_skip", "submit_result", "submit_culprit",
    "read_file_at_parent", "run_repro_with_traceback", "inspect_at", "submit_root_cause",
    "apply_patch", "run_repro", "run_tests", "submit_fix",
    "search_memory", "load_skill", "update_skill",
    # Manager (FSM brain) delegates to the sub-agents via these tools.
    "run_repro_agent", "run_bisect_agent", "run_root_cause_agent",
    "run_fix_agent", "finish",
}

EXPECTED_AGENTS = {"repro", "bisect", "root_cause", "fix", "manager"}


def test_all_tools_discovered():
    assert EXPECTED_TOOLS == set(_FACTORIES)


def test_registry_sees_same_tools_as_tools_module():
    assert set(_TOOL_FACTORIES) == set(_FACTORIES)


def test_tool_factories_are_callable():
    for name, factory in _TOOL_FACTORIES.items():
        assert callable(factory), f"Factory for '{name}' is not callable"


def test_all_agents_in_config():
    registry = AutoDebugRegistry.from_file()
    assert EXPECTED_AGENTS == set(registry.available_agents())


def test_agent_runners_exported():
    runners = {
        "repro": run_repro,
        "bisect": run_bisect,
        "root_cause": run_root_cause,
        "fix": run_fix,
        "manager": run_manager,
    }
    for name, fn in runners.items():
        assert callable(fn), f"{name} runner is not callable"
        assert fn.__name__ == f"run_{name}"
