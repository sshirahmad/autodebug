"""Single registry — discovers tool factories, loads agent config, builds tools.

Tool discovery:
    Any callable exported from autodebug.tools named make_{tool}_tool is
    registered. To add a tool: create the factory, import it in
    autodebug/tools/__init__.py — done.

Agent discovery:
    Agents are plain `run_<name>` functions exported from autodebug.agents
    (see autodebug/agents/__init__.py). Each agent's behavior is configured
    by the matching config/agents/<name>.json file.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import autodebug.tools as _tools_module
from autodebug.config.loader import ConfigLoader
from autodebug.config.schema import AgentConfig, PipelineConfig


def _discover_tools() -> dict[str, Callable]:
    return {
        attr[len("make_"):-len("_tool")]: obj
        for attr, obj in inspect.getmembers(_tools_module, callable)
        if attr.startswith("make_") and attr.endswith("_tool")
    }


_TOOL_FACTORIES = _discover_tools()


class AutoDebugRegistry:
    def __init__(self, config: PipelineConfig, config_dir: str | Path | None = None):
        self.config = config
        self._loader = ConfigLoader(config_dir)

    @classmethod
    def from_file(cls, config_dir: str | Path | None = None) -> "AutoDebugRegistry":
        loader = ConfigLoader(config_dir)
        return cls(loader.load(), config_dir)

    def get_config(self, agent_name: str) -> AgentConfig:
        if agent_name not in self.config.agents:
            raise KeyError(
                f"Agent '{agent_name}' not in config. Available: {sorted(self.config.agents)}"
            )
        return self.config.agents[agent_name]

    def system_prompt(self, agent_name: str) -> str:
        cfg = self.get_config(agent_name)
        base = self._loader.resolve_prompt(cfg.system_prompt)
        skills = self._loader.resolve_skills(cfg.skills)
        return f"{base}\n\n---\n\n{skills}" if skills else base

    def prompt_states(self, agent_name: str) -> dict[str, str]:
        """Return the per-state system prompts for a multi-state agent.

        The agent's `system_prompt` config points at a YAML whose top-level keys
        are state names (e.g. the Manager's FSM phases) mapped to prompts.
        """
        cfg = self.get_config(agent_name)
        return self._loader.resolve_prompt_map(cfg.system_prompt)

    def build_tools(self, agent_name: str, **context) -> list:
        cfg = self.get_config(agent_name)
        ctx = {"agent_name": agent_name, **context}
        tools = []
        for name in cfg.tools:
            if name not in _TOOL_FACTORIES:
                raise KeyError(
                    f"Unknown tool '{name}'. Available: {sorted(_TOOL_FACTORIES)}"
                )
            tools.append(_TOOL_FACTORIES[name](**ctx))
        return tools

    def available_tools(self) -> list[str]:
        return sorted(_TOOL_FACTORIES)

    def available_agents(self) -> list[str]:
        return sorted(self.config.agents)

    def run(self, repo_url: str, bug_report: str, **kwargs):
        from autodebug.graph.pipeline import run_pipeline
        return run_pipeline(repo_url=repo_url, bug_report=bug_report, **kwargs)

    def __repr__(self) -> str:
        return (
            f"AutoDebugRegistry(\n"
            f"  agents={self.available_agents()},\n"
            f"  tools={self.available_tools()}\n"
            f")"
        )
