"""Single registry — discovers tools and agents, assembles the pipeline.

Tool discovery:
    Any callable exported from autodebug.tools named make_{tool}_tool is
    registered. To add a tool: create the factory, import it in
    autodebug/tools/__init__.py — done.

Agent discovery:
    Any BaseAgent subclass exported from autodebug.agents is registered under
    its snake_case name (RootCauseAgent → root_cause). To add an agent: create
    the class, import it in autodebug/agents/__init__.py, add
    config/agents/<name>.json — done.

The registry builds a tool_builder closure from the config tool names and
passes it to agents at creation time. Agents call it with runtime context
(repo_path, sandbox, …) inside their run() method — no tool imports needed
in agent files.

Usage:
    registry = AutoDebugRegistry.from_file()
    print(registry.available_tools())
    print(registry.available_agents())
    result = registry.run(repo_url=..., bug_report=...)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import autodebug.tools as _tools_module
import autodebug.agents as _agents_module
from autodebug.agents.base import BaseAgent
from autodebug.config.loader import ConfigLoader
from autodebug.config.schema import AgentConfig, PipelineConfig


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_tools() -> dict[str, Callable]:
    return {
        attr[len("make_"):-len("_tool")]: obj
        for attr, obj in inspect.getmembers(_tools_module, callable)
        if attr.startswith("make_") and attr.endswith("_tool")
    }


def _discover_agents() -> dict[str, type]:
    found = {}
    for _, cls in inspect.getmembers(_agents_module, inspect.isclass):
        if issubclass(cls, BaseAgent) and cls is not BaseAgent:
            name = cls.__name__.removesuffix("Agent")
            key = "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
            found[key] = cls
    return found


_TOOL_FACTORIES = _discover_tools()
_AGENT_CLASSES  = _discover_agents()


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------

def _make_tool_builder(names: list[str], agent_name: str) -> Callable:
    """Return a callable that builds tool instances when given runtime context."""
    for name in names:
        if name not in _TOOL_FACTORIES:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(_TOOL_FACTORIES)}")

    def build(**context) -> list:
        ctx = {"agent_name": agent_name, **context}
        return [_TOOL_FACTORIES[name](**ctx) for name in names]

    return build


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AutoDebugRegistry:
    def __init__(self, config: PipelineConfig, config_dir: str | Path | None = None):
        self.config = config
        self._loader = ConfigLoader(config_dir)

    @classmethod
    def from_file(cls, config_dir: str | Path | None = None) -> "AutoDebugRegistry":
        loader = ConfigLoader(config_dir)
        return cls(loader.load(), config_dir)

    def available_tools(self) -> list[str]:
        return sorted(_TOOL_FACTORIES)

    def available_agents(self) -> list[str]:
        return sorted(_AGENT_CLASSES)

    def create_agent(self, name: str) -> BaseAgent:
        if name not in _AGENT_CLASSES:
            raise KeyError(f"Unknown agent '{name}'. Available: {self.available_agents()}")
        cfg = self._get(name)
        cls = _AGENT_CLASSES[name]
        base_prompt = self._loader.resolve_prompt(cfg.system_prompt)
        skill_content = self._loader.resolve_skills(cfg.skills)
        system_prompt = (
            f"{base_prompt}\n\n---\n\n{skill_content}" if skill_content else base_prompt
        )
        kwargs: dict = dict(
            system_prompt=system_prompt,
            time_budget_seconds=cfg.time_budget_seconds,
            token_budget=cfg.token_budget,
            cost_budget_usd=cfg.cost_budget_usd,
            cost_per_1k_input_tokens=cfg.cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=cfg.cost_per_1k_output_tokens,
            max_retries=cfg.max_retries,
            **_model_kwargs(cfg),
            **cfg.extra,
        )
        if "tool_builder" in inspect.signature(cls.__init__).parameters:
            kwargs["tool_builder"] = _make_tool_builder(cfg.tools, name)
        return cls(**kwargs)

    def create_repro_agent(self):      return self.create_agent("repro")
    def create_bisect_agent(self):     return self.create_agent("bisect")
    def create_root_cause_agent(self): return self.create_agent("root_cause")
    def create_fix_agent(self):        return self.create_agent("fix")

    def create_pipeline(self):
        from autodebug.graph.pipeline import build_graph
        return build_graph(self)

    def run(self, repo_url: str, bug_report: str, **kwargs):
        from autodebug.graph.pipeline import build_graph
        from autodebug.telemetry import setup_tracing
        from autodebug.state import DebugState
        setup_tracing()
        app = build_graph(self)
        initial_state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
        result = app.invoke(initial_state, config={"configurable": {"thread_id": "main"}})
        from autodebug.graph.pipeline import _coerce
        return _coerce(result)

    def _get(self, key: str) -> AgentConfig:
        if key not in self.config.agents:
            raise KeyError(f"Agent '{key}' not in config. Available: {sorted(self.config.agents)}")
        return self.config.agents[key]

    def __repr__(self) -> str:
        return (
            f"AutoDebugRegistry(\n"
            f"  agents={self.available_agents()},\n"
            f"  tools={self.available_tools()}\n"
            f")"
        )


def _model_kwargs(cfg: AgentConfig) -> dict:
    kwargs = {}
    if cfg.model:
        kwargs["model_id"] = cfg.model
    if cfg.provider:
        kwargs["provider"] = cfg.provider
    return kwargs
