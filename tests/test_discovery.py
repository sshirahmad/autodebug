"""Tests for auto-discovery of tools and agents from __init__ imports."""

from autodebug.tools import _FACTORIES
from autodebug.registry import _TOOL_FACTORIES, _AGENT_CLASSES
from autodebug.agents.repro import ReproAgent
from autodebug.agents.bisect import BisectAgent
from autodebug.agents.root_cause import RootCauseAgent
from autodebug.agents.fix import FixAgent


EXPECTED_TOOLS = {
    "read_file", "list_files",
    "run_script", "submit_repro",
    "mark_bad", "mark_good", "mark_skip", "submit_result",
    "read_file_at_parent", "run_repro_with_traceback", "submit_root_cause",
    "apply_patch", "run_repro", "run_tests", "submit_fix",
}

EXPECTED_AGENTS = {"repro", "bisect", "root_cause", "fix"}


def test_all_tools_discovered():
    from autodebug.tools import _FACTORIES as f
    assert EXPECTED_TOOLS == set(f)


def test_registry_sees_same_tools_as_tools_module():
    from autodebug.tools import _FACTORIES as f
    assert set(_TOOL_FACTORIES) == set(f)


def test_all_agents_discovered():
    assert EXPECTED_AGENTS == set(_AGENT_CLASSES)


def test_agent_classes_are_correct():
    assert _AGENT_CLASSES["repro"] is ReproAgent
    assert _AGENT_CLASSES["bisect"] is BisectAgent
    assert _AGENT_CLASSES["root_cause"] is RootCauseAgent
    assert _AGENT_CLASSES["fix"] is FixAgent


def test_tool_factories_are_callable():
    for name, factory in _TOOL_FACTORIES.items():
        assert callable(factory), f"Factory for '{name}' is not callable"
