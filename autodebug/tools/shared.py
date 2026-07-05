"""Tools shared across multiple agents: read_file, list_files.

Every file operation routes through the long-lived sandbox container so the
repo state stays inside the Docker volume.
"""

from __future__ import annotations

from langchain_core.tools import tool

from autodebug.sandbox import Sandbox


_CONTAINER_PREFIXES = ("/workspace/repo/", "/workspace/repo", "/repo/", "/repo")


def _to_repo_relative(path: str) -> str:
    """Strip container-absolute prefixes the LLM might prepend."""
    for prefix in _CONTAINER_PREFIXES:
        bare = prefix.rstrip("/")
        if path == bare or path.startswith(bare + "/"):
            path = path[len(prefix):] if path.startswith(prefix) else path[len(bare):]
            break
    return path.lstrip("/")


def make_read_file_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
        """Read a file from the repository.

        Args:
            path: Relative path from the repo root.
            offset: Line number to start reading from (0-based). Default 0.
            limit: Maximum number of lines to return. Default 500.
        """
        return sandbox.read_file(_to_repo_relative(path), offset=offset, limit=limit)
    return read_file


def make_list_files_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def list_files(path: str) -> str:
        """List files in a directory of the repository.

        Args:
            path: Relative directory path from the repo root ("" or "." for the root).
        """
        rel = _to_repo_relative(path) if path not in ("", ".") else "."
        return sandbox.list_files(rel or ".")
    return list_files


def make_shell_tool(sandbox: Sandbox, **_):
    @tool(parse_docstring=True)
    def shell(command: str) -> str:
        """Run a shell command in the repo sandbox and return its exit code + output.

        A general-purpose terminal inside the container (cwd = repo root). Use it
        for things the other tools don't cover — inspecting or setting up the
        environment (e.g. `pip install <dep>`, `find . -name '*.pyc' -delete`,
        `ls`, `grep`, running language-specific tooling). It runs in a disposable
        per-run container, so it cannot affect the host; commands are time-limited.

        Args:
            command: The shell command to run (bash), e.g. "pip install regex".
        """
        run = sandbox.exec(command)
        return f"exit_code={run.exit_code}\n{run.output[-4000:]}"
    return shell
