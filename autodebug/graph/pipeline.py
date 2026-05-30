"""Pipeline orchestrator — sequences clone → repro → bisect → root_cause → fix.

Each agent stage is a plain `run_<name>(state, *, registry) -> state` function
built with LangChain's `create_agent` (see autodebug/agents/). No LangGraph
node/edge wiring — short-circuit if any stage marks the state FAILED.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import git

from autodebug.agents import run_bisect, run_fix, run_repro, run_root_cause
from autodebug.memory import store_agent_run
from autodebug.state import DebugState, PipelineStage
from autodebug.telemetry import setup_tracing


# ---------------------------------------------------------------------------
# Clone + symlink repair
# ---------------------------------------------------------------------------

_SYMLINK_PATH_RE = re.compile(r"^[\w./\-]+$")
_SYMLINK_MAX_BYTES = 256


def _looks_like_broken_symlink(fp: Path, repo_root: Path) -> Path | None:
    """If fp is a Git-on-Windows broken symlink (text file containing a relative
    path to another file inside the repo), return the resolved target Path.
    Otherwise return None.
    """
    try:
        if fp.stat().st_size > _SYMLINK_MAX_BYTES:
            return None
        content = fp.read_text(encoding="utf-8").strip()
    except (UnicodeDecodeError, OSError):
        return None

    if not content or "\n" in content or not _SYMLINK_PATH_RE.match(content):
        return None

    target = (fp.parent / content).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError:
        return None
    if not target.is_file() or target == fp.resolve():
        return None
    return target


def _repair_broken_symlinks(repo_root: Path, max_passes: int = 5) -> int:
    """Replace Git-on-Windows broken-symlink text files with their target's content."""
    total = 0
    for _ in range(max_passes):
        repaired_this_pass = 0
        for fp in repo_root.rglob("*"):
            if not fp.is_file() or ".git" in fp.parts:
                continue
            target = _looks_like_broken_symlink(fp, repo_root)
            if target is None:
                continue
            try:
                fp.write_bytes(target.read_bytes())
                repaired_this_pass += 1
            except OSError:
                continue
        total += repaired_this_pass
        if repaired_this_pass == 0:
            break
    return total


def clone_repo(state: DebugState) -> DebugState:
    tmp = tempfile.mkdtemp(prefix="autodebug_")
    repo = git.Repo.clone_from(state.repo_url, tmp)
    if state.pre_fix_commit:
        repo.git.checkout(state.pre_fix_commit)
    _repair_broken_symlinks(Path(tmp))
    state.repo_local_path = tmp
    state.stage = PipelineStage.REPRO
    return state


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_STAGES = (
    ("repro", run_repro),
    ("bisect", run_bisect),
    ("root_cause", run_root_cause),
    ("fix", run_fix),
)


def _failed(state: DebugState) -> bool:
    # state.stage may be enum or its string value depending on coercion path.
    return str(state.stage) in (PipelineStage.FAILED.value, str(PipelineStage.FAILED))


def run_pipeline(repo_url: str, bug_report: str, **kwargs) -> DebugState:
    """Run the full clone → repro → bisect → root_cause → fix sequence."""
    setup_tracing()
    from autodebug.registry import AutoDebugRegistry
    registry = AutoDebugRegistry.from_file()

    state = DebugState(repo_url=repo_url, bug_report=bug_report, **kwargs)
    state = clone_repo(state)

    for name, runner in _STAGES:
        if _failed(state):
            break
        state = runner(state, registry=registry)
        if os.getenv("AUTODEBUG_MEMORY_ENABLED", "0") == "1":
            store_agent_run(name, state)

    return state
