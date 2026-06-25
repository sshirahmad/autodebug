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
        # shared package cache mounted + root prefix set.
        assert runner._CONDA_PKGS_VOLUME in cap["volumes"]
        assert cap["environment"] == {"MAMBA_ROOT_PREFIX": runner._CONDA_ROOT}

    def test_defaults_python_when_version_absent(self, monkeypatch):
        monkeypatch.setenv("AUTODEBUG_DEFAULT_PYTHON", "3.10")
        cap = _capture_clone(monkeypatch)  # no python_version
        assert "python=3.10" in cap["command"][2]

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

    def test_exec_disables_pytest_plugin_autoload(self):
        # A version-skewed plugin (e.g. pytest-timeout on pytest 5.x) auto-loads and
        # crashes collection — so it must never auto-load in the scoring sandbox.
        env = self._exec_env()
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "PYTEST_ADDOPTS" not in env  # nothing force-enabled by default

    def test_exec_reenables_allowlisted_plugins(self, monkeypatch):
        # Async projects can opt a specific plugin back in via `-p`.
        monkeypatch.setenv("AUTODEBUG_PYTEST_PLUGINS", "pytest_asyncio, pytest_trio")
        env = self._exec_env()
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert env["PYTEST_ADDOPTS"] == "-p pytest_asyncio -p pytest_trio"
