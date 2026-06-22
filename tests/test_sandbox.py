"""Tests for sandbox setup — the per-bug conda env wiring (AUTODEBUG_CONDA)."""

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
    def test_conda_env_created_and_pip_routed(self, monkeypatch):
        monkeypatch.setenv("AUTODEBUG_CONDA", "1")
        cap = _capture_clone(monkeypatch, requirements="pluggy==0.13.1",
                             python_version="3.7.3")
        cmd = cap["command"][2]  # ["bash", "-c", <cmd>]
        assert "micromamba create -y -p /workspace/env" in cmd
        assert "python=3.7.3" in cmd
        # pip installs route through the env, not the image pip.
        assert "/workspace/env/bin/pip install" in cmd
        assert "pip install" not in cmd.replace("/workspace/env/bin/pip install", "")
        # shared package cache mounted + root prefix set.
        assert runner._CONDA_PKGS_VOLUME in cap["volumes"]
        assert cap["environment"] == {"MAMBA_ROOT_PREFIX": runner._CONDA_ROOT}
        assert cap["image"] == "autodebug-sandbox-conda:latest"

    def test_legacy_unchanged_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("AUTODEBUG_CONDA", raising=False)
        cap = _capture_clone(monkeypatch, python_version="3.7.3")
        cmd = cap["command"][2]
        assert "micromamba" not in cmd
        assert "pip install --target=" in cmd  # legacy image pip + target dir
        assert runner._CONDA_PKGS_VOLUME not in cap.get("volumes", {})
        assert cap.get("environment") is None

    def test_no_conda_without_python_version(self, monkeypatch):
        monkeypatch.setenv("AUTODEBUG_CONDA", "1")
        cap = _capture_clone(monkeypatch)  # flag on but no version -> legacy path
        assert "micromamba" not in cap["command"][2]
        assert runner._CONDA_PKGS_VOLUME not in cap.get("volumes", {})
