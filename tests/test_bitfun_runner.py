"""Tests for the BitFun benchmark runner (eval/bitfun_runner.py).

No Rust build or Docker needed: the subprocess layer (`produce`) and the oracle
(`validate`) are injected, so we test the plumbing, production fidelity, and result
schema deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import bitfun_runner as br  # noqa: E402


_INSTANCE = {
    "id": "black-1", "project": "black",
    "repo_url": "https://github.com/psf/black",
    "bug_report": "black crashes in AWS Lambda when /dev/shm is unavailable",
    "pre_fix_commit": "abc123~1", "buggy_commit_id": "abc123",
    "fixed_commit_id": "def456",
    "test_command": "python -m unittest -q tests.test_black.X",
    "ground_truth_patch": "diff --git ...",
}


class TestArgv:
    def test_argv_shape_and_flags(self):
        argv = br.bitfun_exec_argv("bitfun-cli", "the message", agent="agentic",
                                   patch_path="/tmp/p.diff", session_id="abc")
        assert argv[:3] == ["bitfun-cli", "exec", "the message"]
        assert "--output-patch" in argv and "/tmp/p.diff" in argv
        assert "--agent" in argv and "agentic" in argv
        assert "--session-id" in argv and "abc" in argv   # so we can query usage after

    def test_autonomous_prompt_wraps_report_and_forbids_questions(self):
        msg = br._autonomous_prompt(_INSTANCE["bug_report"])
        assert _INSTANCE["bug_report"] in msg            # core bug report preserved
        assert "AskUserQuestion" in msg                  # explicitly forbids the blocking tool
        # production fidelity: held-out test/fix data must never leak into the prompt
        for leak in (_INSTANCE["test_command"], _INSTANCE["fixed_commit_id"],
                     _INSTANCE["ground_truth_patch"]):
            assert leak not in msg


class TestUsageParsing:
    def test_parses_last_json_line(self):
        out = "log line\n{\"usage\": {\"total_tokens\": 1234, \"cost\": 0.42}}\n"
        assert br._parse_usage(out) == {"tokens": 1234, "cost": 0.42}

    def test_non_json_returns_none(self):
        assert br._parse_usage("no json here") == {"tokens": None, "cost": None}

    def test_parses_bitfun_usage_markdown(self):
        md = (
            "# Session Usage Report\n\n## Tokens\n\n| Metric | Value |\n| --- | --- |\n"
            "| Source | provider |\n| Input | 12,000 |\n| Output | 3,400 |\n"
            "| Total | 15,400 |\n| Cached | 9,600 (80%) |\n"
        )
        r = br.parse_usage_report(md)
        assert r == {"tokens": 15400, "input_tokens": 12000,
                     "output_tokens": 3400, "cache_hit_rate": 0.8}

    def test_usage_markdown_missing_fields_are_none(self):
        r = br.parse_usage_report("no tokens table here")
        assert r["tokens"] is None and r["cache_hit_rate"] is None

    def test_cost_from_tokens_needs_a_price(self, monkeypatch):
        meta = {"input_tokens": 10000, "output_tokens": 2000}
        monkeypatch.setattr(br, "_COST_PER_1K_IN", 0.0)
        monkeypatch.setattr(br, "_COST_PER_1K_OUT", 0.0)
        assert br._cost_from_tokens(meta) is None            # no price -> None
        monkeypatch.setattr(br, "_COST_PER_1K_IN", 0.003)
        monkeypatch.setattr(br, "_COST_PER_1K_OUT", 0.015)
        assert br._cost_from_tokens(meta) == round(10 * 0.003 + 2 * 0.015, 4)


class TestProducePatch:
    def test_clones_buggy_ref_and_reads_emitted_patch(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, cwd=None, timeout=None):
            calls.append(cmd)
            # when bitfun `exec` runs, it writes the patch file (simulate the agent)
            if cmd[:2] == ["bitfun-cli", "exec"]:
                Path(cmd[cmd.index("--output-patch") + 1]).write_text("PATCH-BODY")
            class _P:
                stdout = '{"usage": {"total_tokens": 5, "cost": 0.01}}'
            return _P()

        patch, meta, root = br.produce_patch(
            _INSTANCE, bitfun_bin="bitfun-cli", workdir_root=tmp_path, run=fake_run)
        assert patch == "PATCH-BODY"
        assert meta["tokens"] == 5 and meta["cost"] == 0.01
        assert "log_tail" in meta and meta["workdir"] == str(tmp_path)
        # cloned then checked out the BUGGY ref (pre_fix_commit)
        assert calls[0][:2] == ["git", "clone"]
        assert any(c[:3] == ["git", "-C", str(tmp_path / "repo")] and "checkout" in c
                   and _INSTANCE["pre_fix_commit"] in c for c in calls)


class TestRunOnInstance:
    def _produce(self, patch, meta=None):
        def _p(instance, **kw):
            return patch, (meta or {"tokens": None, "cost": None}), None
        return _p

    def test_scores_patch_and_returns_comparable_schema(self):
        seen = {}

        def fake_validate(instance, patch):
            seen["patch"] = patch
            return {"category": "fix_pass", "passed": True, "applied": True,
                    "harness_valid": True, "reason": "test passed"}

        res = br.run_bitfun_on_instance(
            _INSTANCE, produce=self._produce("A-PATCH", {"tokens": 9, "cost": 0.03}),
            validate=fake_validate)
        assert seen["patch"] == "A-PATCH"
        assert res["fix_category"] == "fix_pass" and res["fix_success"] is True
        assert res["runner"] == "bitfun"
        assert res["total_tokens"] == 9 and res["total_cost"] == 0.03
        # repro/bisect are N/A for BitFun, not failures
        assert res["repro_success"] is None and res["bisect_correct"] is None
        # same keys the AutoDebug runner emits, so --compare works across runners
        for k in ("instance_id", "project", "fix_category", "fix_validation", "fix_patch"):
            assert k in res

    def test_empty_patch_is_no_patch_without_calling_oracle(self):
        called = {"n": 0}

        def fake_validate(instance, patch):
            called["n"] += 1
            return {}

        res = br.run_bitfun_on_instance(
            _INSTANCE, produce=self._produce("   "), validate=fake_validate)
        assert res["fix_category"] == "no_patch"
        assert called["n"] == 0  # never wastes an oracle run on an empty patch

    def test_produce_failure_is_reported_as_error_not_crash(self):
        def boom(instance, **kw):
            raise RuntimeError("bitfun-cli not found")

        res = br.run_bitfun_on_instance(_INSTANCE, produce=boom, validate=lambda *a: {})
        assert res["fix_category"] == "error"
        assert "bitfun-cli not found" in res["error"]
        assert res["fix_success"] is False
