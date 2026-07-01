"""Tests for eval metric scoring: bisect signals and llm-call tracking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import (  # noqa: E402
    bisect_signals, compute_metrics, validate_fix, _source_files, _HARNESS_ERROR_RE,
    load_partial, _normalize_test_command, compare_runs, calibrate_instance,
)


class TestNormalizeTestCommand:
    def test_tox_is_rewritten_to_pytest(self):
        out = _normalize_test_command("tox tests/test_x.py::test_y")
        assert out == "python -m pytest tests/test_x.py::test_y"

    def test_pytest_command_is_unchanged(self):
        cmd = "pytest tests/test_x.py::test_y"
        assert _normalize_test_command(cmd) == cmd

    def test_py_dot_test_alias_rewritten_to_python_m_pytest(self):
        # Modern pytest drops the `py.test` console entry point -> `command not found`
        # (broke all 10 spacy instances). Rewrite to the entry-point-invariant form.
        assert _normalize_test_command("py.test spacy/tests/test_x.py::test_y") == \
            "python -m pytest spacy/tests/test_x.py::test_y"
        assert _normalize_test_command("py.test-3 a.py").startswith("python -m pytest a.py")

    def test_multiline_tox_each_line_normalized_and_chained(self):
        out = _normalize_test_command("tox tests/a.py::t1\ntox tests/b.py::t2")
        assert out == "python -m pytest tests/a.py::t1 && python -m pytest tests/b.py::t2"

    def test_none_is_empty(self):
        assert _normalize_test_command(None) == ""


class TestSanitizeRequirements:
    def test_drops_editable_and_pkg_resources(self):
        from autodebug.sandbox.runner import _sanitize_requirements
        reqs = (
            "click==7.1.2\n"
            "-e git+https://github.com/x/y@abc#egg=y\n"
            "pkg-resources==0.0.0\n"
            "\n# a comment\n"
            "pluggy==0.13.1\n"
        )
        out = _sanitize_requirements(reqs).splitlines()
        assert out == ["click==7.1.2", "pluggy==0.13.1"]


class TestLoadPartial:
    """Resume support: read an incremental JSONL of per-instance results."""

    def test_missing_file_is_empty(self, tmp_path):
        results, done = load_partial(tmp_path / "nope.jsonl")
        assert results == [] and done == set()

    def test_loads_ids_and_skips_bad_lines(self, tmp_path):
        p = tmp_path / "run.jsonl"
        p.write_text(
            '{"instance_id": "ansible-1", "fix_success": true}\n'
            "\n"                                   # blank line tolerated
            "{ this is not json }\n"               # half-written line tolerated
            '{"instance_id": "ansible-2", "fix_success": false}\n',
            encoding="utf-8",
        )
        results, done = load_partial(p)
        assert done == {"ansible-1", "ansible-2"}
        assert len(results) == 2


class TestCompareRuns:
    """Local experiment-vs-experiment diff: classify per-instance category moves."""

    def _write(self, path, rows):
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def _row(self, iid, cat, repro=True, fix=False):
        return {"instance_id": iid, "fix_category": cat, "fix_success": fix,
                "repro_success": repro, "bisect_correct": False}

    def test_classifies_moves_and_metric_delta(self, tmp_path, capsys):
        a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
        self._write(a, [
            self._row("k-1", "harness_invalid"),
            self._row("k-2", "fix_pass", fix=True),
            self._row("k-3", "harness_invalid"),
        ])
        self._write(b, [
            self._row("k-1", "fix_pass", fix=True),   # newly passing
            self._row("k-2", "fix_fail"),             # lost fix_pass
            self._row("k-3", "no_patch"),             # harness cleared (not to pass)
        ])
        compare_runs(str(a), str(b))
        out = " ".join(capsys.readouterr().out.split())  # collapse column padding
        assert "k-1 harness_invalid -> fix_pass" in out          # newly passing
        assert "k-2 fix_pass -> fix_fail" in out                 # newly failing
        assert "k-3 harness_invalid -> no_patch" in out          # harness cleared
        assert "harness_invalid 2 0 -2" in out                   # count delta 2 -> 0

    def test_reports_ids_only_in_one_file(self, tmp_path, capsys):
        a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
        self._write(a, [self._row("only-a", "fix_pass", fix=True)])
        self._write(b, [self._row("only-b", "fix_fail")])
        compare_runs(str(a), str(b))
        out = " ".join(capsys.readouterr().out.split())
        assert "only in A (1): only-a" in out
        assert "only in B (1): only-b" in out


class TestCalibrateGate:
    """The scoreability gate short-circuits (no Docker) when an instance can't be
    scored at all, so it's recorded as unscoreable up front."""

    def test_no_test_command_is_unscoreable(self):
        r = calibrate_instance({"id": "x-1", "repo_url": "u", "fixed_commit_id": "abc"})
        assert r["scoreable"] is False and "test_command" in r["reason"]
        assert r["instance_id"] == "x-1"

    def test_no_fixed_commit_is_unscoreable(self):
        r = calibrate_instance({"id": "x-1", "repo_url": "u", "test_command": "pytest x"})
        assert r["scoreable"] is False and "fixed_commit" in r["reason"]


class TestValidateFixGuards:
    """validate_fix scores a fix by applying it + the official test in a separate
    sandbox. Without docker we can still verify it short-circuits (scores False)
    before touching the sandbox when there's nothing to validate."""

    def test_empty_patch_is_false(self):
        r = validate_fix({"test_command": "pytest x", "repo_url": "u"}, "")
        assert r["passed"] is False and "no patch" in r["reason"]
        assert r["category"] == "no_patch"
        assert validate_fix({"test_command": "pytest x", "repo_url": "u"}, "   ")["passed"] is False

    def test_missing_test_command_is_false(self):
        # No FAIL_TO_PASS test to validate against -> cannot score a pass. An
        # instance with no test_command is unscoreable, i.e. harness_invalid.
        r = validate_fix({"repo_url": "u"}, "diff --git a/x b/x")
        assert r["passed"] is False and "test_command" in r["reason"]
        assert r["category"] == "harness_invalid"


class TestHarnessErrorClassifier:
    """The fallback heuristic (used only when no gold baseline is available) must
    flag could-not-run signals while ignoring genuine assertion failures."""

    def test_flags_import_and_collection_errors(self):
        assert _HARNESS_ERROR_RE.search("NameError: name 'AioHTTPTestCase' is not defined")
        assert _HARNESS_ERROR_RE.search("ModuleNotFoundError: No module named 'aiohttp'")
        assert _HARNESS_ERROR_RE.search("ImportError: cannot import name 'foo'")
        assert _HARNESS_ERROR_RE.search("collected 0 items")

    def test_ignores_plain_assertion_failure(self):
        out = "AssertionError: 1 != 2\nFAILED tests/test_x.py::test_y - AssertionError"
        assert not _HARNESS_ERROR_RE.search(out)


class TestProductionStateHasNoBenchmarkFields:
    """The production pipeline state must not carry any benchmark/test metadata —
    that lives only in the eval harness."""

    def test_debugstate_excludes_test_metadata(self):
        from autodebug.state import DebugState
        fields = set(DebugState.model_fields)
        leaked = fields & {"test_command", "test_file", "test_patch",
                           "fixed_commit_id", "pre_fix_commit"}
        assert not leaked, f"benchmark metadata leaked into DebugState: {leaked}"


class _Bisect:
    def __init__(self, sha, diff):
        self.culprit_commit = sha
        self.commit_diff = diff


class _State:
    def __init__(self, bisect):
        self.bisect = bisect


def _diff(*paths: str) -> str:
    return "\n".join(f"diff --git a/{p} b/{p}" for p in paths)


class TestSourceFiles:
    def test_excludes_changelog_doc_and_test_paths(self):
        diff = _diff(
            "lib/ansible/cli/galaxy.py",
            "changelogs/fragments/59846.yaml",
            "docs/foo.rst",
            "test/units/cli/test_galaxy.py",
        )
        assert _source_files(diff) == {"lib/ansible/cli/galaxy.py"}


class TestBisectSignals:
    def test_no_bisect_is_all_false(self):
        sig = bisect_signals(_State(None), {"buggy_commit_id": "a"})
        assert sig == {"bisect_sha_match": False, "bisect_file_overlap": False, "bisect_correct": False}

    def test_exact_sha_match(self):
        state = _State(_Bisect("abcdef1234", _diff("unrelated.py")))
        sig = bisect_signals(state, {"buggy_commit_id": "abcdef1234", "ground_truth_patch": ""})
        assert sig["bisect_sha_match"] and sig["bisect_correct"]
        assert not sig["bisect_file_overlap"]

    def test_source_file_overlap_counts(self):
        state = _State(_Bisect("zzz", _diff("lib/ansible/cli/galaxy.py")))
        inst = {"buggy_commit_id": "aaa", "ground_truth_patch": _diff("lib/ansible/cli/galaxy.py")}
        sig = bisect_signals(state, inst)
        assert sig["bisect_file_overlap"] and sig["bisect_correct"]
        assert not sig["bisect_sha_match"]

    def test_changelog_only_overlap_does_not_count(self):
        # The culprit and fix patch share only a changelog file -> no real signal.
        state = _State(_Bisect("zzz", _diff("changelogs/fragments/x.yaml")))
        inst = {
            "buggy_commit_id": "aaa",
            "ground_truth_patch": _diff("lib/ansible/cli/galaxy.py", "changelogs/fragments/x.yaml"),
        }
        sig = bisect_signals(state, inst)
        assert not sig["bisect_file_overlap"]
        assert not sig["bisect_correct"]


class TestComputeMetrics:
    def test_reports_sha_and_overlap_rates_separately(self):
        results = [
            {"repro_success": True, "bisect_correct": True, "bisect_sha_match": False,
             "bisect_file_overlap": True, "fix_success": True},
            {"repro_success": True, "bisect_correct": True, "bisect_sha_match": True,
             "bisect_file_overlap": True, "fix_success": False},
        ]
        m = compute_metrics(results)
        assert m["bisect_accuracy"] == 1.0
        assert m["bisect_sha_match_rate"] == 0.5
        assert m["bisect_file_overlap_rate"] == 1.0
        assert m["fix_rate"] == 0.5

    def test_empty_results_no_zero_division(self):
        assert compute_metrics([]) == {"total": 0}

    def test_harness_invalid_excluded_from_fix_rate(self):
        # A broken-harness instance must NOT count as a fix failure: it's dropped
        # from the denominator and reported separately.
        results = [
            {"repro_success": True, "bisect_correct": False, "fix_success": False,
             "fix_category": "harness_invalid"},
            {"repro_success": True, "bisect_correct": False, "fix_success": True,
             "fix_category": "fix_pass"},
            {"repro_success": True, "bisect_correct": False, "fix_success": False,
             "fix_category": "fix_fail"},
        ]
        m = compute_metrics(results)
        assert m["harness_invalid"] == 1
        assert m["fix_scoreable"] == 2
        assert m["fix_rate"] == 0.5  # 1 pass / 2 scoreable; the invalid one dropped

    def test_all_harness_invalid_is_zero_not_nan(self):
        results = [{"repro_success": False, "bisect_correct": False, "fix_success": False,
                    "fix_category": "harness_invalid"}]
        m = compute_metrics(results)
        assert m["fix_rate"] == 0.0 and m["fix_scoreable"] == 0 and m["harness_invalid"] == 1

    def test_missing_category_treated_as_scoreable(self):
        # Back-compat: results without fix_category (older runs) stay in the denominator.
        results = [
            {"repro_success": True, "bisect_correct": True, "bisect_sha_match": False,
             "bisect_file_overlap": True, "fix_success": True},
            {"repro_success": True, "bisect_correct": True, "bisect_sha_match": True,
             "bisect_file_overlap": True, "fix_success": False},
        ]
        m = compute_metrics(results)
        assert m["fix_rate"] == 0.5 and m["fix_scoreable"] == 2 and m["harness_invalid"] == 0


class TestLlmCallTracking:
    def test_budget_middleware_increments_calls(self):
        from langchain_core.messages import AIMessage
        from autodebug.agents.base import Budget, budget_middleware

        budget = Budget()
        mw = budget_middleware(budget)
        # mw = [_check_budget(before), _track_tokens(after)]; call the after hook.
        msg = AIMessage(content="hi")
        msg.usage_metadata = {"input_tokens": 5, "output_tokens": 3}
        mw[1].after_model(state={"messages": [msg]}, runtime=None)
        mw[1].after_model(state={"messages": [msg]}, runtime=None)
        assert budget.calls == 2
        assert budget.tokens_used == 16
