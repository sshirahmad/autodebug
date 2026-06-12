"""Tests for the runtime-inspection harness builders.

The harnesses are plain Python programs, so we can run them directly with a
subprocess (no Docker) and assert they capture the failure state / probe values.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.tools.introspect import inspect_harness, postmortem_harness  # noqa: E402


def _run(harness: str):
    return subprocess.run([sys.executable, "-c", harness], capture_output=True, text=True)


class TestPostmortemHarness:
    def test_captures_exception_and_frame_locals(self):
        script = "def f(x):\n    y = x + 1\n    raise ValueError('boom %d' % y)\nf(41)\n"
        r = _run(postmortem_harness(script))
        assert r.returncode == 1
        assert "EXCEPTION: ValueError: boom 42" in r.stdout
        assert "x = 41" in r.stdout and "y = 42" in r.stdout      # frame locals dumped
        assert "FRAME STATE (innermost first)" in r.stdout

    def test_crash_frame_comes_first(self):
        # innermost-first ordering: the raising frame (f) appears before <module>
        script = "def f():\n    raise RuntimeError('x')\nf()\n"
        r = _run(postmortem_harness(script))
        assert r.stdout.index(" in f") < r.stdout.index(" in <module>")

    def test_clean_run_produces_no_postmortem(self):
        r = _run(postmortem_harness("x = 1 + 1\n"))
        assert r.returncode == 0
        assert "FRAME STATE" not in r.stdout

    def test_filters_dunder_and_harness_noise(self):
        r = _run(postmortem_harness("raise RuntimeError('x')\n"))
        assert "__name__ =" not in r.stdout and "__loader__ =" not in r.stdout
        assert "_SCRIPT =" not in r.stdout  # harness's own frame skipped


class TestInspectHarness:
    def test_probes_values_at_line_across_hits(self):
        d = tempfile.mkdtemp()
        Path(d, "toy.py").write_text(
            "def g(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
        )
        driver = f"import sys; sys.path.insert(0, {d!r})\nimport toy\ntoy.g(3)\n"
        r = _run(inspect_harness("toy.py", 4, ["i", "total"], driver, max_hits=3))
        assert r.returncode == 0
        # three hits with the loop variable advancing
        assert r.stdout.count('"i"') >= 3
        assert '"total"' in r.stdout and '"locals"' in r.stdout

    def test_no_hits_when_line_not_reached(self):
        d = tempfile.mkdtemp()
        Path(d, "toy2.py").write_text("def h():\n    return 1\n")
        driver = f"import sys; sys.path.insert(0, {d!r})\nimport toy2\ntoy2.h()\n"
        r = _run(inspect_harness("toy2.py", 999, ["x"], driver))
        assert r.stdout.strip() == "[]"
