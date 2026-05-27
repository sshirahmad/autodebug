"""Tests for AutoDebugRegistry — agent creation and pipeline compilation."""

import pytest
from autodebug.registry import AutoDebugRegistry
from autodebug.agents.repro import ReproAgent
from autodebug.agents.bisect import BisectAgent
from autodebug.agents.root_cause import RootCauseAgent
from autodebug.agents.fix import FixAgent
from autodebug.graph import build_graph


def test_registry_loads_from_file():
    registry = AutoDebugRegistry.from_file()
    assert registry is not None


def test_registry_available_agents():
    registry = AutoDebugRegistry.from_file()
    assert set(registry.available_agents()) == {"repro", "bisect", "root_cause", "fix"}


def test_registry_available_tools():
    registry = AutoDebugRegistry.from_file()
    tools = registry.available_tools()
    assert "read_file" in tools
    assert "submit_repro" in tools
    assert "apply_patch" in tools


def test_create_repro_agent():
    registry = AutoDebugRegistry.from_file()
    agent = registry.create_repro_agent()
    assert isinstance(agent, ReproAgent)
    assert callable(agent._tool_builder)


def test_create_bisect_agent():
    registry = AutoDebugRegistry.from_file()
    agent = registry.create_bisect_agent()
    assert isinstance(agent, BisectAgent)


def test_create_root_cause_agent():
    registry = AutoDebugRegistry.from_file()
    agent = registry.create_root_cause_agent()
    assert isinstance(agent, RootCauseAgent)
    assert callable(agent._tool_builder)


def test_create_fix_agent():
    registry = AutoDebugRegistry.from_file()
    agent = registry.create_fix_agent()
    assert isinstance(agent, FixAgent)
    assert callable(agent._tool_builder)


def test_create_agent_by_name():
    registry = AutoDebugRegistry.from_file()
    for name in registry.available_agents():
        agent = registry.create_agent(name)
        assert hasattr(agent, "run"), f"{name}: missing run()"


def test_create_agent_unknown_raises():
    registry = AutoDebugRegistry.from_file()
    with pytest.raises(KeyError, match="Unknown agent"):
        registry.create_agent("does_not_exist")


def test_build_graph_compiles():
    registry = AutoDebugRegistry.from_file()
    graph = build_graph(registry)
    assert graph is not None


def test_tool_builder_produces_tools(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock

    registry = AutoDebugRegistry.from_file()
    agent = registry.create_repro_agent()

    tools = agent._tool_builder(
        repo_path=tmp_path,
        sandbox=MagicMock(),
        result=[],
    )
    tool_names = {t.name for t in tools}
    assert "read_file" in tool_names
    assert "submit_repro" in tool_names
