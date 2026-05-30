"""Tools shared across multiple agents: read_file, list_files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

# Container paths the LLM might use instead of relative paths
_CONTAINER_PREFIXES = ("/workspace/repo/", "/workspace/repo", "/repo/", "/repo")


def _to_repo_relative(path: str) -> str:
    """Strip container-absolute prefixes so paths resolve against repo_path."""
    for prefix in _CONTAINER_PREFIXES:
        if path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/"):
            path = path[len(prefix):]
            break
    return path.lstrip("/")


def make_read_file_tool(repo_path: Path, **_):
    @tool
    def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
        """Read a file from the repository.

        Args:
            path: Relative path from the repo root.
            offset: Line number to start reading from (0-based). Default 0.
            limit: Maximum number of lines to return. Default 500.
        """
        fp = repo_path / _to_repo_relative(path)
        if not fp.exists():
            return f"File not found: {path}"
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)
        chunk = lines[offset: offset + limit]
        header = f"[Lines {offset + 1}-{min(offset + limit, total)} of {total}]\n"
        return header + "".join(chunk)
    return read_file


def make_list_files_tool(repo_path: Path, **_):
    @tool
    def list_files(path: str) -> str:
        """List files in a directory of the repository."""
        dp = repo_path / _to_repo_relative(path)
        if dp.is_dir():
            entries = sorted(dp.iterdir())
            return "\n".join(("  " if e.is_dir() else "") + e.name for e in entries)
        return f"Directory not found: {path}"
    return list_files
