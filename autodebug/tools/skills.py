"""Tools for dynamically loading and updating agent skills."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

_SKILLS_ROOT = Path(__file__).resolve().parents[2] / ".skills"


def make_load_skill_tool(**_):
    @tool
    def load_skill(name: str) -> str:
        """Load a skill's full documentation by name.

        Returns the contents of .skills/<name>/SKILL.md, useful for loading
        extended guidance on-demand (e.g. 'systematic-debugging',
        'git-advanced-workflows', 'investigate').

        Args:
            name: Skill directory name under .skills/ (e.g. 'systematic-debugging').
        """
        skill_file = _SKILLS_ROOT / name / "SKILL.md"
        if not skill_file.exists():
            available = [d.name for d in _SKILLS_ROOT.iterdir() if d.is_dir()] if _SKILLS_ROOT.exists() else []
            return f"Skill '{name}' not found. Available: {available}"
        content = skill_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].lstrip()
        return content
    return load_skill


def make_update_skill_tool(agent_name: str = "", **_):
    @tool
    def update_skill(name: str, content: str, mode: str = "append") -> str:
        """Persist a new or updated skill to .skills/<name>/SKILL.md.

        Use this to capture reusable techniques you discover during a run so
        future agents (and future runs of this agent) can benefit from them.

        Args:
            name: Skill directory name, e.g. 'bisect-tricks' or
                  'git-advanced-workflows'. Will be created if it doesn't exist.
            content: Markdown content to write. Should be a concise, actionable
                     guide — what to do, when, and why.
            mode: 'append' (default) adds content after existing text;
                  'replace' overwrites the entire file.
        """
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return "Error: invalid skill name — must be a plain directory name with no slashes."

        skill_dir = _SKILLS_ROOT / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        if mode == "replace" or not skill_file.exists():
            skill_file.write_text(content.strip() + "\n", encoding="utf-8")
            action = "created" if not skill_file.exists() else "replaced"
        else:
            existing = skill_file.read_text(encoding="utf-8")
            skill_file.write_text(existing.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")
            action = "updated"

        source = f" (written by {agent_name})" if agent_name else ""
        return f"Skill '{name}' {action}{source}. Path: {skill_file}"

    return update_skill
