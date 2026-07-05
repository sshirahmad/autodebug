"""Tools for dynamically loading and updating agent skills."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool

_SKILLS_ROOT = Path(__file__).resolve().parents[2] / ".skills"


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in the same dir + os.replace),
    so a concurrent reader never sees a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-skill-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic on the same filesystem
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def make_load_skill_tool(**_):
    @tool(parse_docstring=True)
    def load_skill(name: str) -> str:
        """Load a skill's full documentation by name.

        Returns the contents of .skills/<name>/SKILL.md, useful for loading
        extended guidance on-demand (e.g. 'investigate', 'bisect-tricks',
        'git-advanced-workflows').

        Args:
            name: Skill directory name under .skills/ (e.g. 'investigate').
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
    @tool(parse_docstring=True)
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
        # Skill writes mutate a single shared host path. During an eval (esp.
        # parallel --workers) that's both a race and a reproducibility hazard —
        # one bug's edit changes another bug's prompt mid-run — so run_eval turns
        # writes off by default. Reads (load_skill) still work.
        if os.getenv("AUTODEBUG_SKILL_WRITES", "1") == "0":
            return ("Skill writes are disabled for this run "
                    "(AUTODEBUG_SKILL_WRITES=0); nothing was persisted.")
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return "Error: invalid skill name — must be a plain directory name with no slashes."

        skill_dir = _SKILLS_ROOT / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        # Cross-process lock so concurrent workers can't lose each other's appends
        # (the append below is a read-modify-write) or read a torn file.
        from filelock import FileLock, Timeout
        try:
            with FileLock(str(skill_dir / ".lock"), timeout=30):
                existed = skill_file.exists()
                if mode == "replace" or not existed:
                    new_text = content.strip() + "\n"
                    action = "replaced" if existed else "created"
                else:
                    existing = skill_file.read_text(encoding="utf-8")
                    new_text = existing.rstrip() + "\n\n" + content.strip() + "\n"
                    action = "updated"
                _atomic_write(skill_file, new_text)
        except Timeout:
            return (f"Skill '{name}' not written: another run holds the lock — "
                    "try again, the change was not persisted.")

        source = f" (written by {agent_name})" if agent_name else ""
        return f"Skill '{name}' {action}{source}. Path: {skill_file}"

    return update_skill
