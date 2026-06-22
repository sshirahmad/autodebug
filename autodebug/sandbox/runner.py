"""Long-lived Docker sandbox attached to a per-pipeline volume.

A single Sandbox owns one container for the lifetime of an agent stage.
The repository lives inside a Docker volume that's shared across all
stages of the pipeline — so symlinks are preserved (Linux ext4 inside
the volume) and there's no per-tool-call container spin-up cost.

Usage:
    sandbox = Sandbox(volume="autodebug_xyz")
    sandbox.start()
    try:
        out = sandbox.run_script("import sys; print(sys.version)")
        text = sandbox.read_file("README.md", offset=0, limit=50)
    finally:
        sandbox.cleanup()
"""

from __future__ import annotations

import base64
import os
import shlex
import uuid
from dataclasses import dataclass

import docker
from docker.errors import ImageNotFound, NotFound


REPO_DIR = "/workspace/repo"

# PYTHONPATH for every sandbox command:
#   - the checked-out repo's source roots FIRST, so the checkout shadows any
#     same-named distribution pre-installed in the image (ansible, black, ...)
#     and stays patchable. Covers flat / src-layout / lib-layout projects.
#   - /workspace/deps LAST: a volume-backed dir where the checkout's own
#     dependencies are installed at clone time (see clone_into_volume). Old
#     checkouts need deps the image lacks — e.g. old black imports `regex`.
_DEPS_DIR = "/workspace/deps"
_SRC_ROOTS = (REPO_DIR, f"{REPO_DIR}/src", f"{REPO_DIR}/lib", _DEPS_DIR)
_PYTHONPATH = ":".join(_SRC_ROOTS)

# Per-bug conda env: at clone time, micromamba builds a Python env at the bug's
# EXACT version ON THE VOLUME, so pip finds matching wheels (old numpy/scipy/etc.
# install instead of failing to build from source) and every stage runs the right
# interpreter. The env is self-contained, so the runtime just prepends its bin to
# PATH. Packages are cached in a shared named volume across bugs (each version +
# wheel downloaded once). The sandbox image is micromamba-based (docker/sandbox).
_CONDA_ENV = "/workspace/env"
_CONDA_PKGS_VOLUME = "autodebug-conda-pkgs"
_CONDA_ROOT = "/opt/mamba"


def _default_python() -> str:
    """Interpreter for bugs that don't record a version (e.g. ad-hoc production
    runs). BugsInPy instances always carry their python_version."""
    return os.getenv("AUTODEBUG_DEFAULT_PYTHON", "3.11")


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


def _b64(s: str) -> str:
    """Base64-encode a string for safe transport through bash."""
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


class Sandbox:
    """Long-lived container that owns the repo and runs commands via docker exec."""

    def __init__(self, volume: str, image: str | None = None, exec_timeout: int | None = None):
        self.volume = volume
        self.image = image or os.getenv("SANDBOX_IMAGE", "autodebug-sandbox:latest")
        # Hard wall-clock cap for EVERY command. Budgets are only checked between
        # LLM turns, so without this a single hanging/slow tool call (e.g. black's
        # ProcessPoolExecutor blocking with no /dev/shm) runs unbounded and blows
        # past every time budget. timeout(1) kills the command at this limit.
        self.exec_timeout = exec_timeout or int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "300"))
        self._client = docker.from_env()
        self._ensure_image()
        self.container = None

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self.image)
        except ImageNotFound:
            raise RuntimeError(
                f"Sandbox image '{self.image}' not found. "
                "Run: docker build -t autodebug-sandbox:latest ./docker/sandbox"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the long-lived container with the shared volume mounted."""
        if self.container is not None:
            return
        self.container = self._client.containers.run(
            image=self.image,
            command=["sleep", "infinity"],
            volumes={self.volume: {"bind": "/workspace", "mode": "rw"}},
            working_dir=REPO_DIR,
            mem_limit=os.getenv("SANDBOX_MEM_LIMIT", "1g"),
            nano_cpus=int(os.getenv("SANDBOX_NANO_CPUS", "2000000000")),
            # Cap process count so a runaway/forkbomb command (now that agents have
            # a general shell) can't exhaust the host; the container is disposable.
            pids_limit=int(os.getenv("SANDBOX_PIDS_LIMIT", "512")),
            network_disabled=False,
            detach=True,
            auto_remove=False,
        )

    def cleanup(self) -> None:
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Primitive: shell exec
    # ------------------------------------------------------------------

    def exec(self, command: str, workdir: str | None = None) -> RunResult:
        """Execute a shell command in the container. workdir defaults to REPO_DIR."""
        if self.container is None:
            raise RuntimeError("Sandbox not started; call start() or use a `with` block")
        # Prepend the per-bug env's bin so `python`/`pip`/`pytest` resolve to the
        # bug's interpreter (the env is built at clone time on the volume).
        command = f'export PATH="{_CONDA_ENV}/bin:$PATH"; {command}'
        # Wrap in `timeout` so no single command can hang the pipeline. SIGTERM at
        # the limit, SIGKILL 10s later; a timed-out command exits 124.
        result = self.container.exec_run(
            cmd=["timeout", "-k", "10", str(self.exec_timeout), "bash", "-c", command],
            workdir=workdir or REPO_DIR,
            demux=True,
            # PYTHONDONTWRITEBYTECODE: never write .pyc. Otherwise the first import
            # caches bytecode and a later apply_patch to the .py is masked by the
            # stale .pyc (Python reuses it when mtimes collide on the volume) — so
            # the fixer's patches silently never execute.
            environment={"PYTHONPATH": _PYTHONPATH, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout_bytes, stderr_bytes = (
            result.output if isinstance(result.output, tuple) else (result.output, b"")
        )
        exit_code = result.exit_code or 0
        stderr = (stderr_bytes or b"").decode(errors="replace")
        if exit_code == 124:
            stderr = (
                f"[command timed out after {self.exec_timeout}s and was killed]\n" + stderr
            )
        return RunResult(
            exit_code=exit_code,
            stdout=(stdout_bytes or b"").decode(errors="replace"),
            stderr=stderr,
        )

    # ------------------------------------------------------------------
    # High-level operations (tools route through these)
    # ------------------------------------------------------------------

    def git(self, *args: str, workdir: str | None = None) -> RunResult:
        """Run a git subcommand inside the repo. e.g. sandbox.git('log', '-1')"""
        joined = " ".join(shlex.quote(a) for a in args)
        return self.exec(f"git {joined}", workdir=workdir or REPO_DIR)

    def run_script(self, script: str, workdir: str | None = None) -> RunResult:
        """Write `script` to /tmp/script.py and run it with python. CWD = repo."""
        encoded = _b64(script)
        cmd = (
            f"echo {encoded} | base64 -d > /tmp/script.py && "
            f"python /tmp/script.py"
        )
        return self.exec(cmd, workdir=workdir or REPO_DIR)

    def read_file(self, path: str, offset: int = 0, limit: int = 500) -> str:
        """Read a slice of a file. offset is 0-based line index."""
        full = f"{REPO_DIR}/{path.lstrip('/')}"
        check = self.exec(
            f"test -f {shlex.quote(full)} && wc -l < {shlex.quote(full)}"
        )
        if check.exit_code != 0:
            return f"File not found: {path}"
        try:
            total = int(check.stdout.strip())
        except ValueError:
            total = 0
        start = max(0, offset) + 1  # sed is 1-based
        end = start + max(1, limit) - 1
        body = self.exec(f"sed -n '{start},{end}p' {shlex.quote(full)}").stdout
        header = f"[Lines {start}-{min(end, total)} of {total}]\n"
        return header + body

    def list_files(self, path: str = ".") -> str:
        """List a directory's entries, marking subdirectories with a leading '  '."""
        full = f"{REPO_DIR}/{path.lstrip('/')}" if path != "." else REPO_DIR
        check = self.exec(f"test -d {shlex.quote(full)}")
        if check.exit_code != 0:
            return f"Directory not found: {path}"
        # ls -A skips . and ..; -p appends / to directories.
        ls = self.exec(f"ls -Ap {shlex.quote(full)}").stdout
        # Format: prefix dirs with two spaces (legacy convention).
        out = []
        for line in ls.splitlines():
            if line.endswith("/"):
                out.append("  " + line.rstrip("/"))
            else:
                out.append(line)
        return "\n".join(out)

    def write_file(self, path: str, content: str) -> RunResult:
        """Overwrite a file in the repo with content."""
        full = f"{REPO_DIR}/{path.lstrip('/')}"
        encoded = _b64(content)
        cmd = (
            f"mkdir -p $(dirname {shlex.quote(full)}) && "
            f"echo {encoded} | base64 -d > {shlex.quote(full)}"
        )
        return self.exec(cmd)


# ----------------------------------------------------------------------
# Volume helpers — used by pipeline.clone_repo
# ----------------------------------------------------------------------

def create_repo_volume() -> str:
    """Create a fresh Docker volume for a pipeline run and return its name."""
    client = docker.from_env()
    name = f"autodebug_{uuid.uuid4().hex[:10]}"
    client.volumes.create(name)
    return name


def remove_repo_volume(name: str) -> None:
    """Remove a previously created repo volume."""
    if not name:
        return
    client = docker.from_env()
    try:
        client.volumes.get(name).remove(force=True)
    except NotFound:
        pass
    except Exception:
        pass


_TEST_PATHSPECS = ("test/", "tests/", "*/test/", "*/tests/", "test_*", "*_test.py")


def _sanitize_requirements(reqs: str) -> str:
    """Keep only installable pins from a frozen requirements blob.

    Drops the self-editable install (the repo is already checked out and on
    PYTHONPATH) and `pkg-resources==0.0.0` (a Debian artifact that breaks pip).
    """
    out = []
    for line in (reqs or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-e ") or s.startswith("-e git+"):
            continue
        if s.lower().startswith("pkg-resources==0.0.0"):
            continue
        out.append(s)
    return "\n".join(out)


def clone_into_volume(
    volume: str,
    repo_url: str,
    ref: str | None,
    *,
    test_patch: str | None = None,
    fixed_commit: str | None = None,
    requirements: str | None = None,
    setup_command: str | None = None,
    python_version: str | None = None,
) -> None:
    """One-shot container that clones repo_url into /workspace/repo on the volume.

    Two ways to sync the test files into a self-consistent state at the
    checkout point:
      - `test_patch`: explicit unified diff (e.g. from SWE-Bench's
        pre-split format) applied via `git apply`.
      - `fixed_commit`: the SHA where the fix landed. The diff between
        the checkout point (`ref`) and `fixed_commit`, restricted to test
        paths, is computed inside the container and applied. Use this for
        benchmarks like BugsInPy where bug_patch.txt only contains the
        source fix and the new tests live in the fix commit itself.

    If both are supplied, `test_patch` wins.
    """
    import base64

    client = docker.from_env()
    image = os.getenv("SANDBOX_IMAGE", "autodebug-sandbox:latest")
    cmd = f"git clone {shlex.quote(repo_url)} {shlex.quote(REPO_DIR)}"
    if ref:
        cmd += f" && cd {shlex.quote(REPO_DIR)} && git checkout {shlex.quote(ref)}"

    # Per-bug conda env at the exact Python version, ON THE VOLUME so every later
    # stage runs the right interpreter and pip finds matching wheels. All pip
    # installs below route through this env.
    cmd += (
        f" && micromamba create -y -p {_CONDA_ENV} -c conda-forge "
        f"python={shlex.quote(python_version or _default_python())} pip"
    )
    pip = f"{_CONDA_ENV}/bin/pip"

    # Install the CHECKED-OUT project's own dependencies into a VOLUME-backed dir
    # so they persist to the long-lived per-stage Sandbox containers (site-packages
    # in this one-shot container is discarded). The image pip-installs the latest
    # version of each subject, but the checkout is usually older with different
    # deps — e.g. old black does `import regex as re` and `regex` isn't present, so
    # `import black` dies before the bug is reached. _DEPS_DIR is on PYTHONPATH.
    if os.getenv("AUTODEBUG_PIP_INSTALL", "1") != "0":
        deps = shlex.quote(_DEPS_DIR)
        # Wrapped in a subshell that always exits 0 so a missing optional dep never
        # aborts the rest of the setup chain (test sync, pyc cleanup).
        cmd += (
            f" && cd {shlex.quote(REPO_DIR)} && ( "
            # 1) the checkout's own RUNTIME deps (the image pins a newer version; the
            #    older checkout often needs different deps — e.g. old black imports `regex`).
            f"(timeout 600 {pip} install --target={deps} . "
            f"|| timeout 300 {pip} install --target={deps} -r requirements.txt || true) ; "
            # 2) TEST/dev deps layered on top. The official FAIL_TO_PASS test routinely
            #    imports test-only deps (pytest plugins, hypothesis, aiohttp for black's
            #    blackd tests, ...). Without them the test MODULE fails to import, which
            #    the eval would otherwise misread as a failed fix rather than a broken
            #    env. Best-effort across the common extra names and dev-requirement files.
            f"for e in d test tests dev testing; do "
            f"timeout 300 {pip} install --target={deps} \".[$e]\" >/dev/null 2>&1 || true; done ; "
            f"for f in test-requirements.txt requirements-test.txt requirements-dev.txt "
            f"dev-requirements.txt requirements/test.txt requirements/dev.txt; do "
            f"if [ -f \"$f\" ]; then timeout 300 {pip} install --target={deps} -r \"$f\" || true; fi; done ; "
            f"true )"
        )
        # 3) Pin the env to the bug's recorded versions, layered ON TOP of the
        #    generic install. Old checkouts need old pytest/pluggy/plugins; the
        #    generic install pulls newer, incompatible ones, so the gold test can't
        #    even run (scored harness_invalid). --upgrade lets the pins REPLACE the
        #    generic versions; --no-deps installs only the frozen set (no newer
        #    transitive pulls). Best-effort: pins that won't build on this image's
        #    Python are skipped, leaving the generic fallback in place.
        sanitized = _sanitize_requirements(requirements or "")
        if sanitized:
            enc_req = base64.b64encode(sanitized.encode("utf-8")).decode("ascii")
            cmd += (
                f" && ( echo {enc_req} | base64 -d > /tmp/reqs.txt ; "
                f"timeout 900 {pip} install --target={deps} --upgrade --no-deps "
                f"-r /tmp/reqs.txt || true )"
            )
        if setup_command and setup_command.strip():
            cmd += f" && ( cd {shlex.quote(REPO_DIR)} && ({setup_command}) || true )"
    # Clear any compiled bytecode so the agents always execute the live source
    # (a stale .pyc would mask their patches).
    cmd += (
        f" && find {shlex.quote(REPO_DIR)} -name '__pycache__' -type d -prune "
        "-exec rm -rf {} + 2>/dev/null || true"
    )

    if test_patch:
        encoded = base64.b64encode(test_patch.encode("utf-8")).decode("ascii")
        cmd += (
            f" && cd {shlex.quote(REPO_DIR)} && "
            f"echo {encoded} | base64 -d | git apply --whitespace=nowarn -"
        )
    elif fixed_commit:
        pathspecs = " ".join(shlex.quote(p) for p in _TEST_PATHSPECS)
        cmd += (
            f" && cd {shlex.quote(REPO_DIR)} && "
            f"git diff HEAD {shlex.quote(fixed_commit)} -- {pathspecs} "
            f"| git apply --whitespace=nowarn - || true"
        )

    client.containers.run(
        image=image,
        command=["bash", "-c", cmd],
        # Share the conda package cache across bugs (a named volume) so each Python
        # version + wheel is downloaded once; the env itself lives on the per-bug
        # repo volume.
        volumes={
            volume: {"bind": "/workspace", "mode": "rw"},
            _CONDA_PKGS_VOLUME: {"bind": _CONDA_ROOT, "mode": "rw"},
        },
        environment={"MAMBA_ROOT_PREFIX": _CONDA_ROOT},
        remove=True,
    )
