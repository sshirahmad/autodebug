"""Tests for AutoDebugRegistry: config loading, tool building, prompt resolution."""

from unittest.mock import MagicMock

import pytest

from autodebug.registry import AutoDebugRegistry


def test_registry_loads_from_file():
    registry = AutoDebugRegistry.from_file()
    assert registry is not None


def test_registry_available_agents():
    registry = AutoDebugRegistry.from_file()
    assert set(registry.available_agents()) == {"repro", "bisect", "root_cause", "fix", "manager"}


def test_registry_available_tools():
    registry = AutoDebugRegistry.from_file()
    tools = registry.available_tools()
    assert "read_file" in tools
    assert "submit_repro" in tools
    assert "apply_patch" in tools


def test_get_config_for_each_agent():
    registry = AutoDebugRegistry.from_file()
    for name in registry.available_agents():
        cfg = registry.get_config(name)
        assert cfg.tools, f"{name}: tools list is empty"
        assert cfg.system_prompt, f"{name}: system_prompt is empty"


def test_get_config_unknown_raises():
    registry = AutoDebugRegistry.from_file()
    with pytest.raises(KeyError, match="not in config"):
        registry.get_config("does_not_exist")


def test_system_prompt_resolves_for_each_agent():
    registry = AutoDebugRegistry.from_file()
    for name in registry.available_agents():
        prompt = registry.system_prompt(name)
        assert len(prompt) > 20, f"{name}: prompt too short ({len(prompt)} chars)"


def test_build_tools_for_repro():
    registry = AutoDebugRegistry.from_file()
    tools = registry.build_tools(
        "repro",
        sandbox=MagicMock(),
        result=[],
    )
    tool_names = {t.name for t in tools}
    assert "read_file" in tool_names
    assert "submit_repro" in tool_names


def test_build_tools_unknown_agent_raises():
    registry = AutoDebugRegistry.from_file()
    with pytest.raises(KeyError, match="not in config"):
        registry.build_tools("does_not_exist", sandbox=MagicMock())
