"""Tools shared across multiple agents: read_file, list_files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool


def make_read_file_tool(repo_path: Path, **_):
    @tool
    def read_file(path: str) -> str:
        """Read a file from the repository."""
        fp = repo_path / path
        if fp.exists():
            return fp.read_text(encoding="utf-8", errors="replace")
        return f"File not found: {path}"
    return read_file


def make_list_files_tool(repo_path: Path, **_):
    @tool
    def list_files(path: str) -> str:
        """List files in a directory of the repository."""
        dp = repo_path / path
        if dp.is_dir():
            entries = sorted(dp.iterdir())
            return "\n".join(("  " if e.is_dir() else "") + e.name for e in entries)
        return f"Directory not found: {path}"
    return list_files
