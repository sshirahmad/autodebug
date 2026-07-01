"""Tests for sandbox setup — the per-bug conda env built at clone time."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.sandbox import runner  # noqa: E402


def _capture_clone(monkeypatch, **kwargs) -> dict:
    """Run clone_into_volume with Docker mocked; return the containers.run kwargs."""
    captured: dict = {}
    client = MagicMock()
    client.containers.run.side_effect = lambda **kw: captured.update(kw)
    monkeypatch.setattr(runner.docker, "from_env", lambda: client)
    runner.clone_into_volume("vol", "https://x/y", "abc", **kwargs)
    return captured


class TestCondaClone:
    def test_env_created_at_recorded_version_and_pip_routed(self, monkeypatch):
        cap = _capture_clone(monkeypatch, requirements="pluggy==0.13.1",
                             python_version="3.7.3")
        cmd = cap["command"][2]  # ["bash", "-c", <cmd>]
        assert "micromamba create -y -p /workspace/env" in cmd
        assert "python=3.7.3" in cmd
        # every pip install routes through the env, never a bare image pip.
        assert "/workspace/env/bin/pip install" in cmd
        assert "pip install" not in cmd.replace("/workspace/env/bin/pip install", "")
        # a baseline test RUNNER is installed (BugsInPy bugs often omit pytest from
        # their runtime freeze), into the env — not a --target dir. But NO plugins:
        # an auto-loaded plugin built for newer pytest crashes collection against an
        # old pinned pytest (broke keras), so plugins come only from the bug's freeze.
        assert "pytest" in cmd
        assert "pytest-asyncio" not in cmd
        assert "--target" not in cmd
        # pinned reqs install per-package (a loop), not one `-r` batch, so one bad
        # pin can't starve later packages (e.g. tensorflow for keras).
        assert "while IFS= read -r req" in cmd
        assert "-r /tmp/reqs.txt" not in cmd
        # A -Werror-stripping compiler shim is installed FIRST on PATH before any build,
        # so the image's modern gcc (15.x) doesn't fatal old code that hardcodes -Werror
        # (pandas tslibs.parsing failed to compile this way -> .so missing -> harness_invalid).
        assert f"{runner._CC_SHIM_DIR}/_wrap" in cmd
        assert f"export PATH={runner._CC_SHIM_DIR}:$PATH" in cmd
        assert "-Werror" in runner._CC_SHIM  # the shim drops it
        # C/Cython extensions are built IN-PLACE (matplotlib ft2font, pandas Cython),
        # else the source on PYTHONPATH shadows the installed copy and import fails.
        assert "setup.py build_ext --inplace" in cmd
        # ...and PARALLEL: a serial -O3 compile of pandas' 50+ Cython extensions blows
        # the build timeout and leaves half the .so missing (-> harness_invalid).
        assert '-j "$(nproc)"' in cmd
        # ...followed by a SERIAL mop-up pass (no --force): the parallel builder is racy
        # and can drop a single straggler's link (observed: only tslibs.parsing missing).
        # The idempotent non-force pass rebuilds just the missing .so, so import succeeds.
        assert cmd.count("setup.py build_ext --inplace") >= 2
        assert "--force" in cmd
        # shared package cache mounted + root prefix set.
        assert runner._CONDA_PKGS_VOLUME in cap["volumes"]
        assert cap["environment"] == {"MAMBA_ROOT_PREFIX": runner._CONDA_ROOT}

    def test_sanitize_requirements_recovers_utf16_mojibake_pins(self):
        # Several BugsInPy freezes were captured as UTF-16 and landed mojibake'd:
        # a NUL byte between every char + a replacement-char BOM. Fed raw to the
        # per-line pip loop they install unreliably (scrapy got old attrs but not the
        # matching old Twisted -> `attr.s(unsafe_hash=…)` TypeError -> all 34 scrapy
        # scored harness_invalid). The sanitizer must strip the noise to clean pins.
        blob = "��" + "\x00".join("Twisted==20.3.0") + "\x00\n" \
               + "\x00".join("attrs==19.3.0") + "\x00"
        out = runner._sanitize_requirements(blob)
        lines = out.splitlines()
        assert "Twisted==20.3.0" in lines
        assert "attrs==19.3.0" in lines
        # no residual NUL / replacement chars survive
        assert "\x00" not in out and "�" not in out

    def test_defaults_python_when_version_absent(self, monkeypatch):
        monkeypatch.setenv("AUTODEBUG_DEFAULT_PYTHON", "3.10")
        cap = _capture_clone(monkeypatch)  # no python_version
        assert "python=3.10" in cap["command"][2]

    def test_test_sync_checks_out_changed_test_files_by_exact_path(self, monkeypatch):
        # The FAIL_TO_PASS test is often ADDED by the fix, so we sync the changed
        # test files to their fixed-commit version. Must use EXACT paths (git
        # checkout's wildcard pathspecs match nothing): diff buggy↔fixed, filter to
        # test files, checkout each exact path from the fixed commit.
        cap = _capture_clone(monkeypatch, fixed_commit="deadbeef1234")
        cmd = cap["command"][2]
        assert "git diff --name-only HEAD deadbeef1234" in cmd
        assert 'git checkout deadbeef1234 -- "$f"' in cmd
        assert "*/tests/" not in cmd  # the broken wildcard-pathspec path is gone

    def test_exec_prepends_env_bin_to_path(self):
        # The runtime resolves python/pip/pytest from the per-bug env.
        sb = runner.Sandbox.__new__(runner.Sandbox)
        sb.container = MagicMock()
        sb.container.exec_run.return_value = MagicMock(output=(b"", b""), exit_code=0)
        sb.exec_timeout = 30
        sb.exec("python --version")
        sent = sb.container.exec_run.call_args.kwargs["cmd"]
        assert 'export PATH="/workspace/env/bin:$PATH";' in sent[-1]

    def _exec_env(self, command="python -m pytest x"):
        sb = runner.Sandbox.__new__(runner.Sandbox)
        sb.container = MagicMock()
        sb.container.exec_run.return_value = MagicMock(output=(b"", b""), exit_code=0)
        sb.exec_timeout = 30
        sb.exec(command)
        return sb.container.exec_run.call_args.kwargs["environment"]

    def test_exec_disables_autoload_and_clears_ini_addopts(self):
        # A version-skewed plugin (e.g. pytest-timeout on pytest 5.x) auto-loads and
        # crashes collection -> autoload off. And a repo pytest.ini's `addopts` can
        # require a now-unloaded plugin (e.g. `-n` from xdist) -> clear addopts.
        env = self._exec_env()
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert env["PYTEST_ADDOPTS"] == "-o addopts= -o filterwarnings="

    def test_exec_reenables_allowlisted_plugins(self, monkeypatch):
        # Async projects can opt a specific plugin back in via `-p`, on top of the
        # addopts/filterwarnings clears.
        monkeypatch.setenv("AUTODEBUG_PYTEST_PLUGINS", "pytest_asyncio, pytest_trio")
        env = self._exec_env()
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert env["PYTEST_ADDOPTS"] == "-o addopts= -o filterwarnings= -p pytest_asyncio -p pytest_trio"
