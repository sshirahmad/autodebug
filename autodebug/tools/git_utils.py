"""Git helpers — all run inside the sandbox container against the shared volume.

There is no host-side git access; every operation goes through `sandbox.git(...)`
so the repo lives entirely inside the Docker volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from autodebug.sandbox import Sandbox


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    diff: str  # full diff vs parent; consumers truncate for display


def get_commit_info(sandbox: Sandbox, sha: str) -> CommitInfo:
    log = sandbox.git("log", "-1", "--format=%H%n%h%n%s%n%an%n%ai", sha)
    lines = log.stdout.strip().splitlines()
    while len(lines) < 5:
        lines.append("")

    diff = sandbox.git("diff", f"{sha}^", sha).stdout
    if not diff.strip():
        # Merge commit or unreachable parent: fall back to `git show`.
        diff = sandbox.git("show", "--format=", sha).stdout

    return CommitInfo(
        sha=lines[0],
        short_sha=lines[1],
        message=lines[2],
        author=lines[3],
        date=lines[4],
        diff=diff or "",
    )


def current_sha(sandbox: Sandbox) -> str:
    return sandbox.git("rev-parse", "HEAD").stdout.strip()


def count_commits_between(sandbox: Sandbox, good: str, bad: str) -> int:
    result = sandbox.git("rev-list", "--count", f"{good}..{bad}")
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def find_commit_before_days(sandbox: Sandbox, days: int) -> str:
    """Return the latest commit SHA that is at least `days` old."""
    result = sandbox.git("rev-list", "-1", f"--before={days} days ago", "HEAD")
    return result.stdout.strip()


def get_file_at_commit(sandbox: Sandbox, sha: str, file_path: str) -> str:
    result = sandbox.git("show", f"{sha}:{file_path}")
    return result.stdout if result.exit_code == 0 else ""


def first_commit(sandbox: Sandbox) -> str:
    """Return the SHA of the very first commit in the repo."""
    result = sandbox.git("rev-list", "--max-parents=0", "HEAD")
    lines = result.stdout.strip().splitlines()
    return lines[0] if lines else ""


def unshallow(sandbox: Sandbox) -> None:
    """Convert a shallow clone to full history (needed for deep bisects)."""
    sandbox.git("fetch", "--unshallow")


def checkout(sandbox: Sandbox, ref: str) -> None:
    sandbox.git("checkout", "--force", ref)


def restore_checkout(sandbox: Sandbox, sha: str) -> None:
    """Return the working tree to `sha`, aborting any in-progress `git bisect`.

    The repo volume is shared across pipeline stages; a crashed or unfinished
    bisect (or a stray checkout) would otherwise leave the tree on the wrong
    commit, so root_cause/fix would run against the wrong code. Best-effort —
    never raises, so it's safe in a `finally`.
    """
    try:
        sandbox.git("bisect", "reset")        # no-op (nonzero) if not bisecting
        sandbox.git("checkout", "--force", sha)
    except Exception:
        pass
