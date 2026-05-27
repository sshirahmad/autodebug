"""Git utilities for bisect and commit inspection."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    diff: str  # diff vs parent


def _git(repo_path: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=check,
    )


def get_commit_info(repo_path: str, sha: str) -> CommitInfo:
    log = _git(repo_path, "log", "-1", "--format=%H%n%h%n%s%n%an%n%ai", sha)
    lines = log.stdout.strip().splitlines()
    diff = _git(repo_path, "diff", f"{sha}^", sha, check=False).stdout
    return CommitInfo(
        sha=lines[0],
        short_sha=lines[1],
        message=lines[2],
        author=lines[3],
        date=lines[4],
        diff=diff[:8000],  # cap diff size sent to LLM
    )


def checkout(repo_path: str, ref: str) -> None:
    _git(repo_path, "checkout", "--force", ref)


def bisect_start(repo_path: str) -> None:
    _git(repo_path, "bisect", "start")


def bisect_reset(repo_path: str) -> None:
    _git(repo_path, "bisect", "reset", check=False)


def bisect_good(repo_path: str, ref: str | None = None) -> str:
    args = ["bisect", "good"] + ([ref] if ref else [])
    result = _git(repo_path, *args)
    return result.stdout.strip()


def bisect_bad(repo_path: str, ref: str | None = None) -> str:
    args = ["bisect", "bad"] + ([ref] if ref else [])
    result = _git(repo_path, *args)
    return result.stdout.strip()


def bisect_skip(repo_path: str) -> str:
    return _git(repo_path, "bisect", "skip").stdout.strip()


def current_sha(repo_path: str) -> str:
    return _git(repo_path, "rev-parse", "HEAD").stdout.strip()


def count_commits_between(repo_path: str, good: str, bad: str) -> int:
    result = _git(repo_path, "rev-list", "--count", f"{good}..{bad}", check=False)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def find_commit_before_days(repo_path: str, days: int) -> str:
    """Return the latest commit SHA that is at least `days` old."""
    result = _git(
        repo_path,
        "rev-list",
        "-1",
        f"--before={days} days ago",
        "HEAD",
        check=False,
    )
    return result.stdout.strip()


def get_file_at_commit(repo_path: str, sha: str, file_path: str) -> str:
    result = _git(repo_path, "show", f"{sha}:{file_path}", check=False)
    return result.stdout if result.returncode == 0 else ""


def first_commit(repo_path: str) -> str:
    """Return the SHA of the very first commit in the repo."""
    result = _git(repo_path, "rev-list", "--max-parents=0", "HEAD", check=False)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def unshallow(repo_path: str) -> None:
    """Convert a shallow clone to full history (needed for deep bisects)."""
    _git(repo_path, "fetch", "--unshallow", check=False)
