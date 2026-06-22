"""Memory recall wiring — the search namespace must reach written memories."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_search_uses_broad_autodebug_prefix(monkeypatch):
    """Memories are written under ('autodebug','manager') in manager mode, so the
    search tool must use the ('autodebug',) PREFIX (not a per-agent sub-namespace)
    or recall finds nothing."""
    from autodebug import memory

    captured = {}

    def fake_create(*, namespace, store, instructions):
        captured["namespace"] = namespace
        m = MagicMock()
        m.invoke.return_value = "ok"
        return m

    monkeypatch.setattr(memory, "memory_store", lambda: object())
    monkeypatch.setattr(memory, "create_search_memory_tool", fake_create)

    for agent in ("repro", "bisect", "root_cause", "fix"):
        memory.make_search_memory_tool(agent_name=agent)
        assert captured["namespace"] == ("autodebug",), agent
