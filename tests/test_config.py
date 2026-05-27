"""Tests for config loading, schema validation, and env-var overrides."""

import pytest
from autodebug.config import load_config
from autodebug.config.schema import PipelineConfig, AgentConfig, SandboxConfig


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)


def test_all_agents_in_config():
    cfg = load_config()
    assert {"repro", "bisect", "root_cause", "fix"} == set(cfg.agents)


def test_agent_config_has_required_fields():
    cfg = load_config()
    for name, agent in cfg.agents.items():
        assert agent.system_prompt, f"{name}: system_prompt is empty"
        assert agent.tools,        f"{name}: tools list is empty"


def test_sandbox_defaults():
    cfg = load_config()
    assert cfg.sandbox.timeout_seconds > 0
    assert cfg.sandbox.image


def test_env_override_sandbox_timeout(monkeypatch):
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "999")
    cfg = load_config()
    assert cfg.sandbox.timeout_seconds == 999



def test_env_override_tracing(monkeypatch):
    monkeypatch.setenv("AUTODEBUG_PHOENIX_ENABLED", "true")
    cfg = load_config()
    assert cfg.tracing.enabled is True


def test_prompt_resolves_from_yaml():
    from autodebug.config.loader import ConfigLoader
    loader = ConfigLoader()
    cfg = loader.load()
    for name, agent in cfg.agents.items():
        prompt = loader.resolve_prompt(agent.system_prompt)
        assert len(prompt) > 20, f"{name}: prompt too short after resolving"
