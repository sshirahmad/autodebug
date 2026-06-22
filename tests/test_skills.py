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
        out = self.loader.resolve_skills(["bisect-tricks"])
        assert "Primary skill (already loaded): bisect-tricks" in out
        # a phrase that only exists in the FULL body, not the 1-line description:
        assert "Match the Repro to the Bug" in out

    def test_secondary_skills_are_descriptions_only(self):
        out = self.loader.resolve_skills(["investigate", "bisect-tricks"])
        # primary fully loaded
        assert "Primary skill (already loaded): investigate" in out
        assert "Step 0 — Anchor on the bug report" in out  # body of investigate
        # secondary listed as description + loadable, NOT its full body
        assert "## Additional skills" in out
        assert "**bisect-tricks**" in out
        assert "Match the Repro to the Bug" not in out  # bisect-tricks body NOT inlined here
        assert "load_skill" in out

    def test_missing_primary_warns_but_does_not_crash(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.loader.resolve_skills(["does-not-exist"])
        assert out == ""  # nothing to inline, no secondary entries


class TestUpdateSkillConcurrencySafety:
    """update_skill must be safe under parallel runs and respect the eval kill-switch."""

    def _tool(self, monkeypatch, tmp_path):
        from autodebug.tools import skills
        monkeypatch.setattr(skills, "_SKILLS_ROOT", tmp_path)
        return skills.make_update_skill_tool(agent_name="t")

    def _call(self, tool, **args):
        out = tool.invoke({"name": tool.name, "args": args, "id": "c1", "type": "tool_call"})
        return getattr(out, "content", None) or str(out)

    def test_disabled_via_env_is_a_noop(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTODEBUG_SKILL_WRITES", "0")
        tool = self._tool(monkeypatch, tmp_path)
        out = self._call(tool, name="s", content="hello")
        assert "disabled" in out.lower()
        assert not (tmp_path / "s" / "SKILL.md").exists()

    def test_create_then_append_under_lock(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTODEBUG_SKILL_WRITES", "1")
        tool = self._tool(monkeypatch, tmp_path)
        self._call(tool, name="s", content="first")
        self._call(tool, name="s", content="second")
        text = (tmp_path / "s" / "SKILL.md").read_text(encoding="utf-8")
        assert "first" in text and "second" in text  # append didn't clobber

    def test_concurrent_appends_dont_lose_writes(self, monkeypatch, tmp_path):
        # Threads share the process; the file lock + atomic write must serialize the
        # read-modify-write so no append is lost.
        import threading
        monkeypatch.setenv("AUTODEBUG_SKILL_WRITES", "1")
        tool = self._tool(monkeypatch, tmp_path)
        self._call(tool, name="s", content="seed")
        def worker(i):
            self._call(tool, name="s", content=f"line-{i}")
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        text = (tmp_path / "s" / "SKILL.md").read_text(encoding="utf-8")
        assert all(f"line-{i}" in text for i in range(8))


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
