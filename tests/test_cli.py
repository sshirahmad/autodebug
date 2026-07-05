"""Tests for the `autodebug debug` CLI — options map into the run config that the
served graph reads (parity with LangGraph Studio's config panel). The graph is
mocked, so no LLM/Docker is exercised."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import autodebug.graph as graph_mod  # noqa: E402
from autodebug import cli  # noqa: E402
from autodebug.state import DebugState  # noqa: E402


def _debug_option_flags() -> set[str]:
    """The option flags registered on the `debug` subcommand. Introspect the click
    command rather than parsing --help text: the rendered help wraps/truncates at
    narrow terminal widths (e.g. in CI), which is flaky."""
    debug_cmd = typer.main.get_command(cli.app).commands["debug"]
    flags: set[str] = set()
    for p in debug_cmd.params:
        flags.update(getattr(p, "opts", []))
    return flags


class _FakeGraph:
    """Captures the config/payload the CLI passes; yields no interrupts."""
    last_config: dict | None = None
    last_payload: dict | None = None

    async def astream(self, payload, config=None, **kw):
        _FakeGraph.last_config = config
        _FakeGraph.last_payload = payload
        return
        yield  # noqa: unreachable — makes this an (empty) async generator

    async def aget_state(self, config):
        return SimpleNamespace(
            values={"debug": DebugState(repo_url="u", bug_report="b").model_dump()})


def _run(monkeypatch, args):
    monkeypatch.setattr(graph_mod, "build_graph", lambda **kw: _FakeGraph())
    return CliRunner().invoke(cli.app, ["debug", *args])


class TestCliParity:
    def test_debug_is_a_named_subcommand(self):
        # `autodebug debug <repo>` must work as documented (Typer would otherwise
        # collapse a single command to `autodebug <repo>`).
        assert "debug" in typer.main.get_command(cli.app).commands

    def test_new_options_are_registered(self):
        flags = _debug_option_flags()
        for opt in ("--ref", "--requirements", "--setup-command", "--python-version"):
            assert opt in flags

    def test_ref_and_env_fields_reach_config(self, monkeypatch):
        r = _run(monkeypatch, ["https://x/y", "--bug", "boom", "--ref", "abc123",
                               "--setup-command", "make build", "--python-version", "3.11"])
        assert r.exit_code == 0, r.output
        cfg = _FakeGraph.last_config["configurable"]
        assert cfg["ref"] == "abc123"
        assert cfg["setup_command"] == "make build"
        assert cfg["python_version"] == "3.11"

    def test_requirements_reads_a_file(self, monkeypatch, tmp_path):
        f = tmp_path / "freeze.txt"
        f.write_text("numpy==1.2.3\nflask==2.0\n")
        r = _run(monkeypatch, ["https://x/y", "--bug", "b", "--requirements", str(f)])
        assert r.exit_code == 0, r.output
        assert _FakeGraph.last_config["configurable"]["requirements"] == "numpy==1.2.3\nflask==2.0\n"

    def test_requirements_inline_text(self, monkeypatch):
        r = _run(monkeypatch, ["https://x/y", "--bug", "b", "--requirements", "flask==2.0"])
        assert r.exit_code == 0, r.output
        assert _FakeGraph.last_config["configurable"]["requirements"] == "flask==2.0"

    def test_issue_only_is_allowed_and_referenced_in_message(self, monkeypatch):
        r = _run(monkeypatch, ["https://x/y", "--issue", "https://github.com/o/r/issues/1"])
        assert r.exit_code == 0, r.output
        assert "issues/1" in str(_FakeGraph.last_payload)

    def test_requires_bug_or_issue(self, monkeypatch):
        import re
        r = _run(monkeypatch, ["https://x/y"])
        assert r.exit_code == 2
        # Strip ANSI + collapse whitespace so a wrapped/boxed error (narrow CI
        # terminal) still matches the phrase.
        clean = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", r.output).split())
        assert "Provide a bug" in clean


class TestCliRendering:
    def test_text_of_handles_str_and_blocks(self):
        assert cli._text_of("plain") == "plain"
        assert cli._text_of([{"type": "text", "text": "a"}, "b"]) == "ab"
        assert cli._text_of(None) == ""

    def test_first_line_collapses_and_truncates(self):
        assert cli._first_line("a\n  b\tc") == "a b c"
        assert cli._first_line("x" * 200, n=10) == "x" * 10 + "…"

    def test_streams_text_but_compacts_tool_output(self, monkeypatch):
        from io import StringIO
        from rich.console import Console
        from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

        stream = [
            ("ns", "messages", (AIMessageChunk(content="hello ", id="1"), {})),
            ("ns", "messages", (AIMessageChunk(content="world", id="1"), {})),
            # a huge tool dump arriving via the message stream must NOT be printed inline
            ("ns", "messages", (ToolMessage(content="HUGE MEMORY DUMP " * 50,
                                            tool_call_id="x", name="search_memory"), {})),
            # the retry middleware's rate-limit notice must be suppressed
            ("ns", "messages", (AIMessage(content="Model call failed after 4 attempts with "
                                          "RateLimitError: 429 free-models-per-day", id="9"), {})),
            # a failed tool result (via updates) shows as a compact one-liner
            ("ns", "updates", {"tools": {"messages": [
                ToolMessage(content="err line1\nline2", tool_call_id="x",
                            name="run_script", status="error")]}}),
        ]

        class _G:
            async def astream(self, payload, config=None, **kw):
                for item in stream:
                    yield item

            async def aget_state(self, config):
                return SimpleNamespace(
                    values={"debug": DebugState(repo_url="u", bug_report="b").model_dump()})

        buf = StringIO()
        monkeypatch.setattr(cli, "console", Console(file=buf, width=200))
        monkeypatch.setattr(graph_mod, "build_graph", lambda **kw: _G())
        r = CliRunner().invoke(cli.app, ["debug", "https://x/y", "--bug", "b"])
        out = buf.getvalue()
        assert r.exit_code == 0, out
        assert "hello world" in out                 # AI text streamed inline
        assert "HUGE MEMORY DUMP" not in out         # tool dump filtered from the text stream
        assert "Model call failed" not in out        # rate-limit retry notice suppressed
        assert "err line1 line2" in out              # failed tool result, collapsed to one line
