"""Tool for dynamically loading skill content during an agent run."""

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
