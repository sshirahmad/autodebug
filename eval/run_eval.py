"""Evaluation harness — runs AutoDebug against a dataset and reports metrics."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a plain script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.graph import run_pipeline
from autodebug.state import PipelineStage


RESULTS_DIR = Path(__file__).parent / "results"

STATUS_OK  = "[ok]"
STATUS_MID = "[~]"
STATUS_ERR = "[x]"


def bisect_correct(state, instance: dict) -> bool:
    if state.bisect is None:
        return False
    truth = instance.get("buggy_commit_id", "")
    found = state.bisect.culprit_commit
    return bool(truth and (found.startswith(truth) or truth.startswith(found)))


def fix_correct(state, instance: dict) -> bool:
    return state.fix is not None and state.stage == PipelineStage.DONE


def run_on_instance(instance: dict) -> dict:
    start = time.time()
    try:
        state = run_pipeline(
            repo_url=instance["repo_url"],
            bug_report=instance["bug_report"],
            known_good_commit=instance.get("known_good_commit"),
        )
        return {
            "instance_id": instance["id"],
            "project": instance.get("project", ""),
            "repro_success": state.repro is not None and state.repro.confirmed,
            "bisect_correct": bisect_correct(state, instance),
            "fix_success": fix_correct(state, instance),
            "stage_reached": state.stage,
            "total_tokens": state.total_tokens,
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
            "bisect_correct": False,
            "fix_success": False,
            "stage_reached": PipelineStage.FAILED,
            "error": str(e),
            "wall_seconds": round(time.time() - start, 1),
        }


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    return {
        "total": n,
        "repro_rate": sum(r["repro_success"] for r in results) / n,
        "bisect_accuracy": sum(r["bisect_correct"] for r in results) / n,
        "fix_rate": sum(r["fix_success"] for r in results) / n,
        "avg_tokens": sum(r.get("total_tokens", 0) for r in results) / n,
        "avg_wall_seconds": sum(r.get("wall_seconds", 0) for r in results) / n,
    }


def main(dataset_path: str, limit: int = 0) -> None:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    instances = dataset[:limit] if limit else dataset

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
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/datasets/buginspy.json"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(dataset_path, limit)
