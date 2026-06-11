"""Tests for skill resolution — the primary (first) skill is inlined in full,
the rest stay as load_skill-on-demand descriptions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.config.loader import ConfigLoader  # noqa: E402
from autodebug.registry import AutoDebugRegistry  # noqa: E402


class TestResolveSkills:
    def setup_method(self):
        self.loader = ConfigLoader()

    def test_empty_returns_blank(self):
        assert self.loader.resolve_skills([]) == ""

    def test_primary_skill_body_is_inlined(self):
        out = self.loader.resolve_skills(["systematic-debugging"])
        assert "Primary skill (already loaded): systematic-debugging" in out
        # a phrase that only exists in the FULL body, not the 1-line description:
        assert "Iron Law" in out

    def test_secondary_skills_are_descriptions_only(self):
        out = self.loader.resolve_skills(["investigate", "systematic-debugging"])
        # primary fully loaded
        assert "Primary skill (already loaded): investigate" in out
        assert "Step 0 — Anchor on the bug report" in out  # body of investigate
        # secondary listed as description + loadable, NOT its full body
        assert "## Additional skills" in out
        assert "**systematic-debugging**" in out
        assert "Iron Law" not in out  # systematic-debugging body NOT inlined here
        assert "load_skill" in out

    def test_missing_primary_warns_but_does_not_crash(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.loader.resolve_skills(["does-not-exist"])
        assert out == ""  # nothing to inline, no secondary entries


class TestSystemPromptCarriesPrimarySkill:
    def test_root_cause_prompt_includes_env_artifact_guard(self):
        # The investigate skill (root_cause's primary) must reach the prompt so the
        # anti-red-herring guidance is always present, even without load_skill.
        sp = AutoDebugRegistry.from_file().system_prompt("root_cause")
        assert "Primary skill (already loaded): investigate" in sp
        assert "environment artifact" in sp.lower()
        assert "ModuleNotFound" in sp

    def test_each_skilled_agent_inlines_its_first_skill(self):
        reg = AutoDebugRegistry.from_file()
        for agent in ("root_cause", "fix", "bisect"):
            cfg = reg.get_config(agent)
            if not cfg.skills:
                continue
            sp = reg.system_prompt(agent)
            assert f"Primary skill (already loaded): {cfg.skills[0]}" in sp
