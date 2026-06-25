"""Tests for token-level streaming (autodebug/agents/base.py).

The forwarder + gating logic is unit-tested without a real model: a model that
streams is provider-dependent, but the contract here — "when enabled and a writer
is live, each token is relayed" — is pure and testable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodebug.agents import base  # noqa: E402


class TestStreamTokenGating:
    def test_off_by_default(self):
        base.set_stream_tokens(False)
        assert base._stream_tokens_active() is False
        assert base._token_forwarder() is None

    def test_contextvar_enables(self, monkeypatch):
        writer = MagicMock()
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: writer)
        base.set_stream_tokens(True)
        try:
            fwd = base._token_forwarder("repro")
            assert isinstance(fwd, base._TokenStreamForwarder)
        finally:
            base.set_stream_tokens(False)

    def test_env_var_enables(self, monkeypatch):
        base.set_stream_tokens(False)
        monkeypatch.setenv("AUTODEBUG_STREAM_TOKENS", "1")
        writer = MagicMock()
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: writer)
        assert base._stream_tokens_active() is True
        assert isinstance(base._token_forwarder(), base._TokenStreamForwarder)

    def test_no_forwarder_without_live_writer(self, monkeypatch):
        # Enabled, but no stream writer in context -> build models unchanged.
        base.set_stream_tokens(True)
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: None)
        try:
            assert base._token_forwarder() is None
        finally:
            base.set_stream_tokens(False)


class TestForwarder:
    def test_relays_each_token_to_writer(self):
        writer = MagicMock()
        fwd = base._TokenStreamForwarder(writer, agent="fix")
        fwd.on_llm_new_token("Hel")
        fwd.on_llm_new_token("lo")
        assert writer.call_count == 2
        assert writer.call_args_list[0].args[0] == {"token": "Hel", "agent": "fix"}

    def test_empty_token_is_ignored(self):
        writer = MagicMock()
        base._TokenStreamForwarder(writer).on_llm_new_token("")
        writer.assert_not_called()

    def test_writer_failure_is_swallowed(self):
        writer = MagicMock(side_effect=RuntimeError("no consumer"))
        # Must not raise — streaming is decorative.
        base._TokenStreamForwarder(writer).on_llm_new_token("x")
