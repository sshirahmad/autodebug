"""Load and assemble PipelineConfig from the config/ directory."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from autodebug.config.schema import AgentConfig, PipelineConfig, SandboxConfig, TracingConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


class ConfigLoader:
    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else _CONFIG_DIR

    def load(self) -> PipelineConfig:
        pipeline_raw = self._read_json("pipeline.json")
        agents = self._load_agents()

        config = PipelineConfig(
            sandbox=SandboxConfig(**pipeline_raw.get("sandbox", {})),
            tracing=TracingConfig(**pipeline_raw.get("tracing", {})),
            agents=agents,
        )
        return self._apply_env_overrides(config)

    def resolve_prompt(self, path: str) -> str:
        """Load the `system` key from a per-agent YAML file in config/prompts/."""
        yaml_path = self.config_dir.parent / path
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return data["system"].strip()

    def _read_json(self, relative: str) -> dict:
        path = self.config_dir / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_agents(self) -> dict[str, AgentConfig]:
        result = {}
        d = self.config_dir / "agents"
        if not d.is_dir():
            return result
        for fp in sorted(d.glob("*.json")):
            data = json.loads(fp.read_text(encoding="utf-8"))
            extra = {k: v for k, v in data.items() if k not in AgentConfig.model_fields}
            result[fp.stem] = AgentConfig(
                system_prompt=data["system_prompt"],
                tools=data["tools"],
                extra=extra,
            )
        return result

    def _apply_env_overrides(self, config: PipelineConfig) -> PipelineConfig:
        data = config.model_dump()

        if val := os.getenv("SANDBOX_IMAGE"):
            data["sandbox"]["image"] = val
        if val := os.getenv("SANDBOX_TIMEOUT_SECONDS"):
            data["sandbox"]["timeout_seconds"] = int(val)
        if val := os.getenv("AUTODEBUG_PHOENIX_ENABLED"):
            data["tracing"]["enabled"] = val.lower() == "true"
        if val := os.getenv("AUTODEBUG_PHOENIX_ENDPOINT"):
            data["tracing"]["phoenix_endpoint"] = val
        return PipelineConfig(**data)


def load_config(config_dir: str | Path | None = None) -> PipelineConfig:
    return ConfigLoader(config_dir).load()
