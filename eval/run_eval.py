"""Evaluation harness — runs AutoDebug against a dataset and reports metrics."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a plain script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.graph import run_pipeline
from autodebug.patch_utils import is_test_path
from autodebug.state import PipelineStage


RESULTS_DIR = Path(__file__).parent / "results"

STATUS_OK  = "[ok]"
STATUS_MID = "[~]"
STATUS_ERR = "[x]"


_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)
# Paths that carry no signal about *which commit introduced the bug* — a
# changelog/doc/test overlap should not make a bisect look correct.
_NOISE_DIR_RE = re.compile(r"(^|/)(changelogs?|docs?|examples?)(/|$)")


def _source_files(diff: str) -> set[str]:
    """Files a diff touches, excluding changelog/doc/test noise."""
    files = set(_DIFF_FILE_RE.findall(diff or ""))
    return {f for f in files if not _NOISE_DIR_RE.search(f) and not is_test_path(f)}


def bisect_signals(state, instance: dict) -> dict:
    """Score bisect, reporting the exact-SHA match and the file-overlap heuristic
    separately so an inflated `bisect_correct` is visible at a glance.

    - sha_match: the submitted SHA matches `buggy_commit_id` (the strict signal).
    - file_overlap: the culprit's diff and the fix patch touch a common *source*
      file (changelog/doc/test paths excluded). This accepts "the commit that
      introduced the buggy code" — what our prompt asks for — when it differs from
      BugsInPy's checkout-reference SHA, without crediting noise-only overlaps.
    - correct: sha_match OR file_overlap (kept as the headline metric).
    """
    if state.bisect is None:
        return {"bisect_sha_match": False, "bisect_file_overlap": False, "bisect_correct": False}

    truth = instance.get("buggy_commit_id", "")
    found = state.bisect.culprit_commit or ""
    sha_match = bool(truth and (found.startswith(truth) or truth.startswith(found)))

    fix_files = _source_files(instance.get("ground_truth_patch", ""))
    culprit_files = _source_files(state.bisect.commit_diff or "")
    file_overlap = bool(fix_files and culprit_files and (fix_files & culprit_files))

    return {
        "bisect_sha_match": sha_match,
        "bisect_file_overlap": file_overlap,
        "bisect_correct": sha_match or file_overlap,
    }


def validate_fix(instance: dict, patch: str) -> dict:
    """Independently score a fix and return diagnostics.

    The agents never see the test (production fidelity), so scoring happens here:
      1. clone the repo at the buggy commit and sync the official test files,
      2. apply the agent's source patch,
      3. run the FAIL_TO_PASS test command — pass = real fix.

    Returns a dict: {passed, applied, reason, test_output} so a 0% fix rate is
    explainable (patch wouldn't apply vs. test failed vs. no patch) without
    digging through traces.
    """
    test_command = instance.get("test_command")
    if not patch or not patch.strip():
        return {"passed": False, "applied": False, "reason": "no patch produced", "test_output": ""}
    if not test_command:
        return {"passed": False, "applied": False, "reason": "instance has no test_command", "test_output": ""}

    import base64
    from autodebug.sandbox import (
        Sandbox, clone_into_volume, create_repo_volume, remove_repo_volume,
    )

    # git apply requires a trailing newline — without it the patch is rejected as
    # "corrupt patch at line N" (the last line). Normalize before applying.
    patch = patch.rstrip("\n") + "\n"

    volume = create_repo_volume()
    try:
        clone_into_volume(
            volume,
            instance["repo_url"],
            instance.get("pre_fix_commit"),
            test_patch=instance.get("test_patch"),
            fixed_commit=instance.get("fixed_commit_id"),
        )
        with Sandbox(volume=volume) as sb:
            encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
            applied = sb.exec(f"echo {encoded} | base64 -d | git apply --whitespace=nowarn -")
            if applied.exit_code != 0:
                applied = sb.exec(f"echo {encoded} | base64 -d | git apply --whitespace=nowarn --3way -")
            if applied.exit_code != 0:
                return {"passed": False, "applied": False,
                        "reason": f"patch did not apply: {applied.stderr[-300:]}", "test_output": ""}
            run = sb.exec(test_command)
            return {
                "passed": run.exit_code == 0,
                "applied": True,
                "reason": "test passed" if run.exit_code == 0 else "patch applied but test failed",
                "test_output": run.output[-1500:],
            }
    except Exception as e:
        return {"passed": False, "applied": False, "reason": f"validation error: {e}", "test_output": ""}
    finally:
        remove_repo_volume(volume)


def fix_validation(state, instance: dict) -> dict:
    """Validate the agent's fix; returns the diagnostics dict from validate_fix."""
    if state.fix is None:
        return {"passed": False, "applied": False, "reason": "no fix submitted", "test_output": ""}
    return validate_fix(instance, state.fix.patch)


def run_on_instance(instance: dict) -> dict:
    start = time.time()
    try:
        # Production-real inputs only — the agents must not receive any benchmark
        # test metadata. `ref` is the buggy checkout point; test_command / test_patch /
        # fixed_commit_id are used solely by validate_fix() in fix_correct() below.
        state = run_pipeline(
            repo_url=instance["repo_url"],
            bug_report=instance["bug_report"],
            ref=instance.get("pre_fix_commit"),
            known_good_commit=instance.get("known_good_commit"),
        )
        fix_diag = fix_validation(state, instance)
        return {
            "instance_id": instance["id"],
            "project": instance.get("project", ""),
            "repro_success": state.repro is not None and state.repro.confirmed,
            **bisect_signals(state, instance),
            "fix_success": fix_diag["passed"],
            "fix_validation": fix_diag,  # why a fix passed/failed (applied? test output)
            "fix_patch": (state.fix.patch[:4000] if state.fix else ""),
            "stage_reached": state.stage,
            "total_tokens": state.total_tokens,
            "total_cost": round(state.total_cost, 4),
            "llm_calls": state.total_llm_calls,
            "wall_seconds": round(time.time() - start, 1),
            "error": state.error,
            "ground_truth_buggy_commit": instance.get("buggy_commit_id", ""),
            "ground_truth_patch_len": len(instance.get("ground_truth_patch", "")),
        }
    except Exception as e:
        return {
            "instance_id": instance["id"],
            "project": instance.get("project", ""),
            "repro_success": False,
            "bisect_sha_match": False,
            "bisect_file_overlap": False,
            "bisect_correct": False,
            "fix_success": False,
            "stage_reached": PipelineStage.FAILED,
            "error": str(e),
            "wall_seconds": round(time.time() - start, 1),
        }


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {"total": 0}
    return {
        "total": n,
        "repro_rate": sum(r["repro_success"] for r in results) / n,
        "bisect_accuracy": sum(r["bisect_correct"] for r in results) / n,
        "bisect_sha_match_rate": sum(r.get("bisect_sha_match", False) for r in results) / n,
        "bisect_file_overlap_rate": sum(r.get("bisect_file_overlap", False) for r in results) / n,
        "fix_rate": sum(r["fix_success"] for r in results) / n,
        "avg_tokens": sum(r.get("total_tokens", 0) for r in results) / n,
        "avg_cost_usd": sum(r.get("total_cost", 0) for r in results) / n,
        "avg_wall_seconds": sum(r.get("wall_seconds", 0) for r in results) / n,
    }


def select_instances(dataset: list[dict], limit: int = 0, ids: str = "") -> list[dict]:
    """Pick which instances to run.

    `--ids` (comma-separated) takes precedence over `--limit`: it selects those
    exact instances by their `id`, in the order given. Otherwise `limit` keeps
    the first N (0 = all).
    """
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        by_id = {str(inst["id"]): inst for inst in dataset}
        selected = [by_id[w] for w in wanted if w in by_id]
        missing = [w for w in wanted if w not in by_id]
        if missing:
            print(f"warning: {len(missing)} id(s) not found: {', '.join(missing)}")
        return selected
    return dataset[:limit] if limit else dataset


def main(dataset_path: str, limit: int = 0, ids: str = "") -> None:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    instances = select_instances(dataset, limit, ids)
    if not instances:
        print("No instances selected — nothing to run.")
        return

    results = []
    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {instance['id']} ...", end=" ", flush=True)
        result = run_on_instance(instance)
        results.append(result)
        if result["fix_success"]:
            status = STATUS_OK
        elif result["repro_success"]:
            status = STATUS_MID
        else:
            status = STATUS_ERR
        print(status)

    metrics = compute_metrics(results)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float) and v <= 1:
            print(f"  {k}: {v:.1%}")
        else:
            print(f"  {k}: {v}")

    out_path = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoDebug against an eval dataset.")
    parser.add_argument(
        "dataset", nargs="?", default="eval/datasets/buginspy.json",
        help="Path to the dataset JSON (default: eval/datasets/buginspy.json).",
    )
    parser.add_argument(
        "limit", nargs="?", type=int, default=0,
        help="Run only the first N instances (0 = all). Ignored when --ids is given.",
    )
    parser.add_argument(
        "--ids", default="",
        help="Comma-separated instance ids to run (overrides limit), e.g. --ids pandas:23,black:4.",
    )
    args = parser.parse_args()
    main(args.dataset, args.limit, args.ids)
