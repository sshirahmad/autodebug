"""Benchmark BitFun's Code Agent on BugsInPy, scored by AutoDebug's held-out oracle.

This is the "swap only the agent" half of a direct AutoDebug-vs-BitFun comparison:
we drive BitFun's CLI (`bitfun exec … --output-patch`, which emits a `git diff` of
the workspace — its built-in SWE-bench evaluation mode) on the *same* buggy checkout,
with the *same* production input (only the bug report — no test metadata), and score
the resulting patch with the *same* `validate_fix` gold-baseline oracle. So the only
variable between the two systems is the agent itself.

BitFun's Code Agent has no separate reproduce/bisect stages, so `repro_success` and
the `bisect_*` fields are reported as N/A (None); only `fix_category` is comparable.

Configuration (env, so the parallel harness inherits it):
  BITFUN_CLI      path to the built `bitfun-cli` binary (default: "bitfun-cli" on PATH)
  BITFUN_AGENT    agent type passed to `--agent` (default: "agentic")
  BITFUN_TIMEOUT  per-instance wall budget in seconds (default: 1800)
  BITFUN_DOCKER_IMAGE  when set (e.g. "autodebug-sandbox:latest"), every bitfun-cli
                  invocation runs inside a `docker run` of this image instead of on
                  the host: the binary, the instance workdir, and ~/.config/bitfun +
                  ~/.bitfun are mounted in, and the container runs as the host uid.
                  Filesystem/process isolation for the agent; network stays open
                  (the model API needs it). BITFUN_DOCKER_CPUS / BITFUN_DOCKER_MEM
                  cap resources (default 2 / 4g).

Standalone smoke (the "run BitFun on one instance" check):
  python -m eval.bitfun_runner --id black-1
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

_DEFAULT_BIN = os.getenv("BITFUN_CLI", "bitfun-cli")
_DEFAULT_AGENT = os.getenv("BITFUN_AGENT", "agentic")
_DEFAULT_TIMEOUT = int(os.getenv("BITFUN_TIMEOUT", "1800"))
# Optional per-1k-token prices to turn BitFun's token counts into a $ cost (BitFun
# reports tokens, not dollars). Set both to the SAME model's price AutoDebug uses.
_COST_PER_1K_IN = float(os.getenv("BITFUN_COST_PER_1K_INPUT", "0") or 0)
_COST_PER_1K_OUT = float(os.getenv("BITFUN_COST_PER_1K_OUTPUT", "0") or 0)
# When set, bitfun-cli runs inside this Docker image (sandboxing the agent's shell/
# edit tools away from the host) instead of directly on the host. Empty = host exec.
_DOCKER_IMAGE = os.getenv("BITFUN_DOCKER_IMAGE", "")


def _autonomous_prompt(bug_report: str) -> str:
    """Wrap the bug report with a non-interactive directive.

    BitFun's `agentic` agent has the `AskUserQuestion` tool and, in one-shot `exec`
    (no interactive client), calling it blocks forever waiting for an answer. For a
    headless batch run we must tell it to act autonomously. The bug report itself is
    unchanged — this only adds the same "no human in the loop, attempt it yourself"
    framing AutoDebug's agents already operate under (fairness preserved).
    """
    return (
        "You are fixing a bug in this repository, running fully autonomously with NO "
        "human available to answer questions. Do NOT ask clarifying questions and do "
        "NOT call AskUserQuestion — make your best attempt and edit the code directly "
        "to fix the issue described below.\n\nBug report:\n" + bug_report
    )


def _cost_from_tokens(meta: dict):
    """USD cost = input/1k*price + output/1k*price, IF prices were configured.

    BitFun reports tokens but not dollars; supply BITFUN_COST_PER_1K_INPUT/OUTPUT
    (the SAME model AutoDebug uses) to make cost directly comparable. Returns None if
    no price is set or token counts are missing."""
    if not (_COST_PER_1K_IN or _COST_PER_1K_OUT):
        return None
    inp, out = meta.get("input_tokens"), meta.get("output_tokens")
    if inp is None and out is None:
        return None
    return round((inp or 0) / 1000 * _COST_PER_1K_IN + (out or 0) / 1000 * _COST_PER_1K_OUT, 4)


def _supports_auto_flag(bitfun_bin: str) -> bool:
    """Whether this bitfun-cli build has `exec --auto` (auto-approve tool requests).

    Newer builds REQUIRE it headless — without it every ExecCommand/Edit is
    permission-rejected and the run ends patchless; older builds don't have the
    flag and never blocked. Probe `exec --help` once per binary path.
    """
    cached = _AUTO_FLAG_CACHE.get(bitfun_bin)
    if cached is not None:
        return cached
    probe = [bitfun_bin, "exec", "--help"]
    if _DOCKER_IMAGE:  # binary may only be runnable inside the container
        probe = _docker_argv(probe, mounts=_docker_mounts(bitfun_bin))
    try:
        proc = subprocess.run(probe, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=60)
        supported = "--auto" in (proc.stdout or "") + (proc.stderr or "")
    except OSError:
        supported = False
    _AUTO_FLAG_CACHE[bitfun_bin] = supported
    return supported


_AUTO_FLAG_CACHE: dict[str, bool] = {}


def _docker_mounts(bitfun_bin: str, root=None) -> list[tuple[str, str, str]]:
    """(host, container, mode) mounts for a containerized bitfun-cli invocation.

    The binary lands on PATH; the user's real config (API key) and session store
    (~/.bitfun, needed so a follow-up `bitfun usage` sees the session) are mounted
    under /bfhome, which HOME points at inside the container. The instance workdir
    (when given) mounts at its own host path so `--output-patch` and -w stay valid.
    """
    binary = shutil.which(bitfun_bin) or bitfun_bin
    home = Path.home()
    mounts = [
        (str(Path(binary).resolve()), "/usr/local/bin/bitfun-cli", "ro"),
        (str(home / ".config" / "bitfun"), "/bfhome/.config/bitfun", "rw"),
        (str(home / ".bitfun"), "/bfhome/.bitfun", "rw"),
    ]
    if root is not None:
        # scratch HOME parent so stray writes (~/.cache, …) succeed under --user
        mounts.insert(0, (str(Path(root) / "home"), "/bfhome", "rw"))
        mounts.append((str(root), str(root), "rw"))
    return mounts


def _docker_argv(inner: list[str], *, mounts, workdir=None, name=None) -> list[str]:
    """Wrap a bitfun-cli argv in `docker run` (image = BITFUN_DOCKER_IMAGE).

    Runs as the host uid/gid (a root-owned file in a bind-mounted $HOME would be
    undeletable for a non-root user) with HOME=/bfhome so bitfun resolves its config
    there. `name` makes the container killable on timeout — killing the `docker run`
    client alone would leak it.
    """
    argv = ["docker", "run", "--rm"]
    if name:
        argv += ["--name", name]
    if hasattr(os, "getuid"):  # not on Windows; Docker Desktop maps perms itself
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["-e", "HOME=/bfhome", "-e", "XDG_CONFIG_HOME=/bfhome/.config"]
    for src, dst, mode in mounts:
        argv += ["-v", f"{src}:{dst}:{mode}"]
    if workdir:
        argv += ["-w", str(workdir)]
    argv += ["--cpus", os.getenv("BITFUN_DOCKER_CPUS", "2"),
             "--memory", os.getenv("BITFUN_DOCKER_MEM", "4g"),
             _DOCKER_IMAGE, "bitfun-cli"]
    return argv + inner[1:]  # inner[0] is the host binary path; in-container it's on PATH


def bitfun_exec_argv(bitfun_bin: str, message: str, *, agent: str, patch_path,
                     session_id: str | None = None, auto: bool | None = None) -> list[str]:
    """The argv for one non-interactive BitFun run that emits a git-diff patch.

    Only the bug-report `message` is passed — never a test command, fixed commit, or
    expected diff — so BitFun sees exactly what AutoDebug's agents see (production
    fidelity). `--output-patch` writes `git diff` of the workspace; `--auto`
    (when the build supports it) approves tool requests so a headless run is never
    permission-rejected. A fixed `--session-id` lets us query `bitfun usage <id>`
    afterwards for token accounting.
    """
    argv = [
        bitfun_bin, "exec", message,
        "--agent", agent,
        "--output-patch", str(patch_path),
        "--output-format", "json",
    ]
    if auto is None:
        auto = _supports_auto_flag(bitfun_bin)
    if auto:
        argv.append("--auto")
    if session_id:
        argv += ["--session-id", session_id]
    return argv


def parse_usage_report(markdown: str) -> dict:
    """Extract token metrics from `bitfun usage <id>` markdown (the ## Tokens table).

    BitFun reports Input/Output/Total tokens + a cache hit-rate (its "token economy"
    metric), but no dollar cost — so `total_cost` stays None unless a price is supplied
    (see run_bitfun_on_instance). Missing/`not reported` fields come back as None.
    """
    def _num(label: str):
        m = re.search(rf"\|\s*{label}\s*\|\s*([\d,]+)\b", markdown or "")
        return int(m.group(1).replace(",", "")) if m else None

    cache = re.search(r"\|\s*Cached\s*\|\s*[\d,]+\s*\((\d+)%\)", markdown or "")
    return {
        "tokens": _num("Total"),
        "input_tokens": _num("Input"),
        "output_tokens": _num("Output"),
        "cache_hit_rate": (int(cache.group(1)) / 100.0) if cache else None,
    }


def _run(cmd: list[str], *, cwd=None, timeout=None):
    """Thin subprocess wrapper (isolated so tests can monkeypatch it).

    Forces UTF-8 decoding: BitFun emits UTF-8 (em-dashes, box-drawing, …) but on
    Windows `text=True` defaults to cp1252, which crashes the pipe reader thread.
    """
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:  # WinError 2 — name the missing executable
        raise FileNotFoundError(
            f"executable not found: {cmd[0]!r}. "
            + ("Set BITFUN_CLI to the full path of bitfun-cli.exe."
               if "bitfun" in cmd[0].lower()
               else "Is it installed and on PATH? (git is required)")
        ) from e


def _parse_usage(stdout: str) -> dict:
    """Best-effort token/cost from BitFun's `--output-format json` (last JSON line)."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        usage = data.get("usage") or {}
        return {"tokens": usage.get("total_tokens"), "cost": usage.get("cost")}
    return {"tokens": None, "cost": None}


def produce_patch(instance: dict, *, bitfun_bin=_DEFAULT_BIN, agent=_DEFAULT_AGENT,
                  timeout=_DEFAULT_TIMEOUT, workdir_root=None, run=_run):
    """Clone the buggy checkout, run BitFun's agent on the bug report, return the patch.

    Returns (patch_text, usage_meta, workdir). The caller owns cleanup of `workdir`.
    """
    root = Path(workdir_root or tempfile.mkdtemp(prefix="bitfun_"))
    repo_dir = root / "repo"
    patch_path = root / "patch.diff"
    ref = instance.get("pre_fix_commit") or instance.get("buggy_commit_id")

    run(["git", "clone", "--quiet", instance["repo_url"], str(repo_dir)], timeout=timeout)
    if ref:
        run(["git", "-C", str(repo_dir), "checkout", "--quiet", ref], timeout=300)

    session_id = uuid.uuid4().hex
    argv = bitfun_exec_argv(bitfun_bin, _autonomous_prompt(instance["bug_report"]),
                            agent=agent, patch_path=patch_path, session_id=session_id)
    container = None
    if _DOCKER_IMAGE:
        (root / "home").mkdir(exist_ok=True)  # scratch HOME for the non-root container
        container = f"bitfun-{session_id[:12]}"
        argv = _docker_argv(argv, mounts=_docker_mounts(bitfun_bin, root=root),
                            workdir=repo_dir, name=container)
    timed_out = False
    try:
        proc = run(argv, cwd=str(repo_dir), timeout=timeout)
        stdout, stderr = getattr(proc, "stdout", "") or "", getattr(proc, "stderr", "") or ""
    except subprocess.TimeoutExpired as e:
        # Salvage partial output so a timeout is diagnosable (and any patch already
        # written to disk is still read below).
        timed_out = True
        stdout = (e.stdout if isinstance(e.stdout, str) else "") or ""
        stderr = ((e.stderr if isinstance(e.stderr, str) else "") or "") + f"\n[TIMED OUT after {timeout}s]"
        if container:
            # killing the `docker run` client doesn't stop the container — kill it
            # by name or the agent keeps running (and billing) unsupervised
            try:
                subprocess.run(["docker", "rm", "-f", container],
                               capture_output=True, timeout=60)
            except Exception:  # noqa: BLE001
                pass

    # Always capture BitFun's own output so a `no_patch`/error is diagnosable.
    out = stdout + "\n----- STDERR -----\n" + stderr
    try:
        (root / "bitfun.log").write_text(out, encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    meta = _parse_usage(stdout)

    # Token accounting: BitFun streams events (no usage total), so query
    # `bitfun usage <session_id>` IN THE WORKSPACE and parse its markdown. Best-effort.
    try:
        usage_argv = [bitfun_bin, "usage", session_id]
        if _DOCKER_IMAGE:  # same mounts: the session store lives in ~/.bitfun
            usage_argv = _docker_argv(usage_argv,
                                      mounts=_docker_mounts(bitfun_bin, root=root),
                                      workdir=repo_dir)
        up = run(usage_argv, cwd=str(repo_dir), timeout=120)
        report = parse_usage_report(getattr(up, "stdout", "") or "")
        if any(v is not None for v in report.values()):
            meta.update(report)
    except Exception:  # noqa: BLE001 — usage accounting must never fail the run
        pass

    meta["log_tail"] = out[-600:]
    meta["workdir"] = str(root)
    meta["timed_out"] = timed_out
    return patch, meta, root


def run_bitfun_on_instance(instance: dict, *, bitfun_bin=_DEFAULT_BIN, agent=_DEFAULT_AGENT,
                           timeout=_DEFAULT_TIMEOUT, keep_workdir=False,
                           validate=None, produce=None) -> dict:
    """Run BitFun on one BugsInPy instance and score its patch with AutoDebug's oracle.

    Returns the SAME result dict shape as `run_on_instance`, so both runners write a
    comparable JSONL and `--compare` works across them. `validate`/`produce` are
    injectable for testing.
    """
    from eval.run_eval import validate_fix, _diag  # reuse the exact oracle + shapes

    validate = validate or validate_fix
    produce = produce or produce_patch
    start = time.time()
    root = None
    try:
        patch, meta, root = produce(instance, bitfun_bin=bitfun_bin, agent=agent, timeout=timeout)
        if patch and patch.strip():
            fix_diag = validate(instance, patch)
        else:
            tail = (meta.get("log_tail") or "").strip()
            why = "BitFun timed out with no patch. " if meta.get("timed_out") else "BitFun produced no patch. "
            fix_diag = _diag("no_patch", passed=False, applied=False, harness_valid=None,
                             reason=why + (f"Output tail: …{tail[-400:]}" if tail else ""))
            if meta.get("workdir"):
                fix_diag["bitfun_workdir"] = meta["workdir"]
        return {
            "instance_id": instance["id"],
            "project": instance.get("project", ""),
            "runner": "bitfun",
            # BitFun has no separate reproduce/bisect stage — N/A, not a failure.
            "repro_success": None,
            "bisect_sha_match": None, "bisect_file_overlap": None, "bisect_correct": None,
            "fix_success": fix_diag["passed"],
            "fix_category": fix_diag.get("category"),
            "harness_valid": fix_diag.get("harness_valid"),
            "fix_validation": fix_diag,
            "fix_patch": (patch or "")[:4000],
            "stage_reached": "done",
            "total_tokens": meta.get("tokens"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "cache_hit_rate": meta.get("cache_hit_rate"),   # BitFun's token-economy metric
            "total_cost": _cost_from_tokens(meta) if meta.get("cost") is None else meta.get("cost"),
            "llm_calls": None,
            "wall_seconds": round(time.time() - start, 1),
            "error": None,
            "ground_truth_buggy_commit": instance.get("buggy_commit_id", ""),
            "ground_truth_patch_len": len(instance.get("ground_truth_patch", "")),
        }
    except Exception as e:  # noqa: BLE001 — one bad instance must not abort the run
        return {
            "instance_id": instance["id"], "project": instance.get("project", ""),
            "runner": "bitfun", "repro_success": None, "bisect_correct": None,
            "fix_success": False, "fix_category": "error",
            "wall_seconds": round(time.time() - start, 1),
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }
    finally:
        if root and not keep_workdir:
            shutil.rmtree(root, ignore_errors=True)


def main(argv=None) -> int:
    """Smoke: run BitFun on a single instance and print its verdict."""
    import argparse

    ap = argparse.ArgumentParser(description="Run BitFun on one BugsInPy instance (smoke).")
    ap.add_argument("--id", required=True, help="Instance id, e.g. black-1")
    ap.add_argument("--dataset", default="eval/datasets/buginspy.json")
    ap.add_argument("--bin", default=_DEFAULT_BIN, help="Path to the bitfun-cli binary")
    ap.add_argument("--agent", default=_DEFAULT_AGENT)
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    ap.add_argument("--keep-workdir", action="store_true")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("instances") or data.get("data") or list(data.values())[0])
    inst = next((x for x in items if str(x.get("id")) == a.id), None)
    if inst is None:
        print(f"instance {a.id!r} not found in {a.dataset}", file=sys.stderr)
        return 2

    res = run_bitfun_on_instance(inst, bitfun_bin=a.bin, agent=a.agent,
                                 timeout=a.timeout, keep_workdir=a.keep_workdir)
    print(json.dumps({k: res.get(k) for k in
                      ("instance_id", "runner", "fix_category", "fix_success",
                       "total_cost", "wall_seconds", "error")}, indent=2))
    return 0 if res.get("fix_category") not in ("error", None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
