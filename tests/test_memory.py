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

    monkeypatch.setenv("AUTODEBUG_MEMORY_ENABLED", "1")  # recall must be on
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


def test_search_tool_is_inert_when_recall_disabled(monkeypatch):
    """With recall off (default), the tool exists but never touches the store."""
    from autodebug import memory

    monkeypatch.delenv("AUTODEBUG_MEMORY_ENABLED", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(memory, "memory_store", lambda: called.__setitem__("n", 1))
    monkeypatch.setattr(memory, "create_search_memory_tool",
                        lambda **k: called.__setitem__("n", 1))

    tool = memory.make_search_memory_tool(agent_name="fix")
    out = tool.invoke({"name": tool.name, "args": {"query": "x"},
                       "id": "c1", "type": "tool_call"})
    assert "disabled" in getattr(out, "content", str(out)).lower()
    assert called["n"] == 0  # store / underlying tool never built
