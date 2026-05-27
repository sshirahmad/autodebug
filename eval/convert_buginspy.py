"""Convert BugsInPy repo into AutoDebug eval dataset (eval/datasets/buginspy.json).

Usage:
    python eval/convert_buginspy.py <path_to_BugsInPy> [--limit N] [--project pandas]

Each output record has everything AutoDebug needs to run and evaluate:
  - repo_url, buggy_commit_id, fixed_commit_id  → ground truth for bisect + fix eval
  - test_command                                 → oracle for repro eval
  - ground_truth_patch                           → oracle for fix eval
  - bug_report                                   → input to the pipeline (from fix commit msg)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OUTPUT_PATH = Path(__file__).parent / "datasets" / "buginspy.json"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_info_file(path: Path) -> dict[str, str]:
    """Parse KEY="VALUE" formatted .info files."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"')
    return result


def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""


def extract_test_command(run_test_sh: str) -> str:
    """Pull the pytest command out of run_test.sh."""
    for line in run_test_sh.splitlines():
        line = line.strip()
        if line.startswith("pytest") or line.startswith("python -m pytest"):
            return line
    return run_test_sh  # fallback: return the whole script


# ---------------------------------------------------------------------------
# GitHub commit message fetch
# ---------------------------------------------------------------------------

def fetch_commit_message(github_url: str, commit_sha: str) -> str:
    """Fetch commit message from GitHub API. Returns empty string on failure."""
    # Parse owner/repo from github_url
    match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", github_url)
    if not match:
        return ""

    repo = match.group(1)
    api_url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["commit"]["message"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return ""


def make_synthetic_report(project: str, bug_id: int, test_command: str) -> str:
    """Fallback bug report when we can't fetch the commit message."""
    return (
        f"Bug in {project} (bug #{bug_id}). "
        f"The following test fails: `{test_command}`. "
        "Please reproduce and fix the issue."
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(buginspy_root: Path, project_filter: str | None, limit: int, fetch_msgs: bool) -> list[dict]:
    records: list[dict] = []

    projects_dir = buginspy_root / "projects"
    project_dirs = sorted(projects_dir.iterdir())

    for proj_dir in project_dirs:
        if not proj_dir.is_dir():
            continue

        project_name = proj_dir.name

        if project_filter and project_name.lower() != project_filter.lower():
            continue

        proj_info = parse_info_file(proj_dir / "project.info")
        if proj_info.get("status", "").upper() != "OK":
            continue

        github_url = proj_info.get("github_url", "")
        bugs_dir = proj_dir / "bugs"

        if not bugs_dir.exists():
            continue

        for bug_dir in sorted(
            (p for p in bugs_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            if not bug_dir.is_dir():
                continue

            bug_id = int(bug_dir.name)
            bug_info = parse_info_file(bug_dir / "bug.info")

            buggy_commit = bug_info.get("buggy_commit_id", "")
            fixed_commit = bug_info.get("fixed_commit_id", "")
            python_version = bug_info.get("python_version", "")
            test_file = bug_info.get("test_file", "")

            run_test = read_file_safe(bug_dir / "run_test.sh")
            setup_sh = read_file_safe(bug_dir / "setup.sh")
            requirements = read_file_safe(bug_dir / "requirements.txt")
            ground_truth_patch = read_file_safe(bug_dir / "bug_patch.txt")
            test_command = extract_test_command(run_test)

            # Build bug report from fixed commit message (most informative signal)
            bug_report = ""
            if fetch_msgs and fixed_commit and github_url:
                bug_report = fetch_commit_message(github_url, fixed_commit)
                time.sleep(0.1)  # stay within GitHub rate limits

            if not bug_report:
                bug_report = make_synthetic_report(project_name, bug_id, test_command)

            records.append({
                "id": f"{project_name}-{bug_id}",
                "project": project_name,
                "bug_id": bug_id,
                "repo_url": github_url,
                "python_version": python_version,
                # Ground truth — used by eval harness, NOT passed to the pipeline
                "buggy_commit_id": buggy_commit,
                "fixed_commit_id": fixed_commit,
                "ground_truth_patch": ground_truth_patch,
                "test_file": test_file,
                "test_command": test_command,
                "setup_command": setup_sh,
                "requirements": requirements,
                # Pipeline inputs
                "bug_report": bug_report,
                "pre_fix_commit": f"{fixed_commit}~1" if fixed_commit else None,
                "known_good_commit": None,  # let BisectAgent find it heuristically
            })

            if limit and len(records) >= limit:
                return records

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert BugsInPy to AutoDebug eval dataset")
    parser.add_argument("buginspy_path", help="Path to cloned BugsInPy repo")
    parser.add_argument("--limit", type=int, default=0, help="Max bugs to include (0=all)")
    parser.add_argument("--project", default="", help="Only include one project (e.g. pandas)")
    parser.add_argument(
        "--fetch-commit-msgs",
        action="store_true",
        help="Fetch fix commit messages from GitHub API (requires GITHUB_TOKEN for best rate limits)",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    buginspy_root = Path(args.buginspy_path).resolve()
    if not (buginspy_root / "projects").exists():
        print(f"ERROR: {buginspy_root} does not look like a BugsInPy repo (no 'projects' dir)")
        raise SystemExit(1)

    print(f"Converting BugsInPy from: {buginspy_root}")
    if args.fetch_commit_msgs:
        token_status = "with token" if GITHUB_TOKEN else "no token — rate limited to 60 req/hr"
        print(f"Fetching commit messages from GitHub ({token_status})")

    records = convert(
        buginspy_root=buginspy_root,
        project_filter=args.project or None,
        limit=args.limit,
        fetch_msgs=args.fetch_commit_msgs,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Print summary
    projects = {}
    for r in records:
        projects[r["project"]] = projects.get(r["project"], 0) + 1

    print(f"\nWrote {len(records)} bugs to {out_path}")
    print("\nBreakdown:")
    for proj, count in sorted(projects.items()):
        print(f"  {proj}: {count}")


if __name__ == "__main__":
    main()
