"""Tests for eval metric scoring: bisect signals and llm-call tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import bisect_signals, compute_metrics, validate_fix, _source_files  # noqa: E402


class TestValidateFixGuards:
    """validate_fix scores a fix by applying it + the official test in a separate
    sandbox. Without docker we can still verify it short-circuits (scores False)
    before touching the sandbox when there's nothing to validate."""

    def test_empty_patch_is_false(self):
        r = validate_fix({"test_command": "pytest x", "repo_url": "u"}, "")
        assert r["passed"] is False and "no patch" in r["reason"]
        assert validate_fix({"test_command": "pytest x", "repo_url": "u"}, "   ")["passed"] is False

    def test_missing_test_command_is_false(self):
        # No FAIL_TO_PASS test to validate against -> cannot score a pass.
        r = validate_fix({"repo_url": "u"}, "diff --git a/x b/x")
        assert r["passed"] is False and "test_command" in r["reason"]


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
