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

    def resolve_prompt_map(self, path: str) -> dict[str, str]:
        """Load every top-level key from a prompt YAML as name -> prompt.

        Used by multi-state agents (e.g. the Manager FSM) where each phase has
        its own system prompt keyed by the phase name.
        """
        yaml_path = self.config_dir.parent / path
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return {
            k: (v.strip() if isinstance(v, str) else v) for k, v in data.items()
        }

    def _load_skill_body(self, name: str) -> str:
        """Return a skill's SKILL.md content with the YAML frontmatter stripped.

        Mirrors the load_skill tool so the auto-injected primary skill reads the
        same as if the agent had loaded it. Empty string if the skill is missing.
        """
        skill_file = _REPO_ROOT / ".skills" / name / "SKILL.md"
        if not skill_file.exists():
            return ""
        content = skill_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].lstrip()
        return content.strip()

    def resolve_skills(self, names: list[str]) -> str:
        """Assemble the skills section of a system prompt.

        Agents almost never call `load_skill` on their own, so the *first*
        configured skill (the agent's primary method) is inlined in FULL here —
        guaranteeing the core guidance is present every run. The remaining skills
        stay as one-line descriptions, loadable on demand via `load_skill`.
        """
        if not names:
            return ""
        skills_root = _REPO_ROOT / ".skills"
        parts: list[str] = []

        primary = names[0]
        primary_body = self._load_skill_body(primary)
        if primary_body:
            parts.append(
                f"## Primary skill (already loaded): {primary}\n\n{primary_body}"
            )
        elif (skills_root / primary / "SKILL.md").exists() is False:
            import warnings
            warnings.warn(f"Skill '{primary}' not found", stacklevel=2)

        entries: list[str] = []
        for name in names[1:]:
            skill_file = skills_root / name / "SKILL.md"
            if not skill_file.exists():
                import warnings
                warnings.warn(f"Skill '{name}' not found at {skill_file}", stacklevel=2)
                continue
            description = _parse_skill_description(skill_file.read_text(encoding="utf-8"))
            entries.append(f"- **{name}**: {description}")
        if entries:
            lines = ["## Additional skills", ""]
            lines.extend(entries)
            lines += ["", "Call `load_skill(name)` to load the full guide for any skill above."]
            parts.append("\n".join(lines))

        return "\n\n---\n\n".join(parts)

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
            # Split: known model fields go directly to AgentConfig, everything
            # else lands in `extra` so the registry can forward it to agents.
            known = {k: v for k, v in data.items() if k in AgentConfig.model_fields and k != "extra"}
            extra = {k: v for k, v in data.items() if k not in AgentConfig.model_fields}
            result[fp.stem] = AgentConfig(extra=extra, **known)
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


def _parse_skill_description(text: str) -> str:
    """Extract the `description` field from YAML frontmatter, or return a fallback."""
    if not text.startswith("---"):
        return "(no description — add a `description:` field to this skill's frontmatter)"
    end = text.find("---", 3)
    if end == -1:
        return "(malformed frontmatter)"
    frontmatter = text[3:end]
    for line in frontmatter.splitlines():
        if line.startswith("description:"):
            return line[len("description:"):].strip().strip('"').strip("'")
    return "(no description — add a `description:` field to this skill's frontmatter)"
