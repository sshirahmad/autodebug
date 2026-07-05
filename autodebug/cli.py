"""CLI entry point for AutoDebug.

One command, one graph. `autodebug debug <repo> --bug "…"` runs the single graph
(autodebug/graph/interactive.py) with HITL on — it streams progress and pauses to
ask you when it gets stuck (a stage exhausts its retries, or the session budget runs
out). `--unattended` turns the prompts off for CI/batch. Same graph the eval harness
and Studio use; the only difference is the `hitl` toggle.
"""

import asyncio
import uuid
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

app = typer.Typer(help="AutoDebug — reproduce, bisect, and fix bugs automatically")
console = Console()


@app.callback()
def _main() -> None:
    """AutoDebug — reproduce, bisect, and fix bugs automatically.

    A no-op callback so Typer keeps `debug` as a *named* subcommand (a single
    command would otherwise collapse to `autodebug <repo>`), matching the docs
    `autodebug debug <repo>` and leaving room for future subcommands.
    """


def _text_of(content) -> str:
    """Printable text from a message content (a str, or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "".join(out)
    return ""


def _first_line(s, n: int = 100) -> str:
    """A compact one-line preview of a (possibly huge, multi-line) tool result."""
    s = " ".join(str(s).split())
    return s[:n] + ("…" if len(s) > n else "")


def _print_summary(debug: dict | None) -> None:
    from autodebug.state import DebugState

    ds = DebugState(**(debug or {"repo_url": "", "bug_report": ""}))
    console.print()  # newline after streamed output
    if ds.repro and ds.repro.confirmed:
        console.print("[green]✓ Bug reproduced[/green]")
    if ds.bisect and ds.bisect.culprit_commit:
        console.print(f"[green]✓ Culprit: {ds.bisect.culprit_commit[:8]}[/green] "
                      f"{ds.bisect.commit_message[:80]}")
    if ds.root_cause:
        console.print(f"[green]✓ Root cause:[/green] {ds.root_cause.summary}")
    if ds.fix:
        console.print("[green]✓ Fix verified[/green]")
        if ds.fix.pr_url:
            console.print(f"  PR: {ds.fix.pr_url}")
    elif ds.error:
        console.print(f"[red]✗ {ds.error}[/red]")
    console.print(f"[dim]LLM calls: {ds.total_llm_calls} | tokens: {ds.total_tokens:,} "
                  f"| ${ds.total_cost:.2f}[/dim]")


@app.command()
def debug(
    repo_url: str = typer.Argument(..., help="GitHub repo URL to debug"),
    bug_report: Optional[str] = typer.Option(None, "--bug", "-b",
                                             help="Bug report text (or use --issue)"),
    issue_url: Optional[str] = typer.Option(None, "--issue", "-i",
                                            help="GitHub issue/PR URL; its title+body is folded into the report"),
    known_good: Optional[str] = typer.Option(None, "--good", "-g",
                                             help="Known good commit hash or date"),
    ref: Optional[str] = typer.Option(None, "--ref", "-r",
                                      help="Commit SHA or branch to check out (default: repository HEAD)"),
    requirements: Optional[str] = typer.Option(None, "--requirements",
                                               help="Pinned deps to layer on top: a path to a "
                                                    "requirements/freeze file, or inline text"),
    setup_command: Optional[str] = typer.Option(None, "--setup-command",
                                                help="Extra shell command run in the repo after dependency install"),
    python_version: Optional[str] = typer.Option(None, "--python-version",
                                                 help="Python version for the sandbox env (e.g. 3.11)"),
    unattended: bool = typer.Option(
        False, "--unattended",
        help="Never pause for input (CI/batch). By default AutoDebug pauses to ask you "
             "when a stage gets stuck or the budget runs out."),
):
    """Reproduce, bisect, root-cause, and fix a bug — pausing to ask for guidance when
    stuck (use --unattended to disable prompts)."""
    from pathlib import Path

    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    from autodebug.graph import build_graph

    if not bug_report and not issue_url:
        raise typer.BadParameter("Provide a bug report with --bug and/or --issue.")
    # --requirements accepts a path to a requirements/freeze file OR inline text.
    reqs = requirements
    if requirements and Path(requirements).is_file():
        reqs = Path(requirements).read_text(encoding="utf-8", errors="replace")

    console.print(Panel(f"[bold blue]AutoDebug[/bold blue]\n{repo_url}", expand=False))
    graph = build_graph(hitl=not unattended)
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "repo_url": repo_url,
                               "issue_url": issue_url, "known_good": known_good,
                               "ref": ref, "requirements": reqs,
                               "setup_command": setup_command,
                               "python_version": python_version}}

    async def _drive(payload):
        # subgraphs=True so the Manager subgraph's live output streams and its
        # interrupts surface. `messages` streams the agents' TEXT token-by-token;
        # `updates` carries tool activity + interrupts. We render them separately so
        # the terminal stays readable: only the model's own text streams inline, while
        # tool calls/results (incl. huge memory JSON, `ls` dumps) print as compact
        # one-liners — otherwise everything blends into one wall of text.
        from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

        current = {"id": None}
        async for _ns, mode, chunk in graph.astream(
            payload, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            if mode == "messages":
                msg, _meta = chunk
                # Only the model's own text — skip tool dumps, HumanMessage nudges, etc.
                if not isinstance(msg, (AIMessage, AIMessageChunk)):
                    continue
                text = _text_of(getattr(msg, "content", ""))
                if not text:
                    continue
                # Suppress the retry middleware's degraded "Model call failed after N
                # attempts with …" notices — huge 429/rate-limit JSON blobs, not agent
                # output. The agent still sees them internally; this is display-only.
                if text.lstrip().startswith("Model call failed after"):
                    continue
                mid = getattr(msg, "id", None)
                if mid != current["id"]:          # blank line between turns
                    console.print()
                    current["id"] = mid
                console.print(text, end="", soft_wrap=True, markup=False)
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    intr = chunk["__interrupt__"]
                    first = intr[0] if isinstance(intr, (list, tuple)) else intr
                    return getattr(first, "value", first)     # paused for input
                for _node, val in (chunk or {}).items():
                    for m in (val.get("messages") if isinstance(val, dict) else None) or []:
                        if isinstance(m, AIMessage):
                            for tc in getattr(m, "tool_calls", None) or []:
                                console.print(f"\n🔧 {tc.get('name', '?')}",
                                              style="cyan", markup=False)
                        elif isinstance(m, ToolMessage):
                            err = getattr(m, "status", None) == "error"
                            console.print(f"   {'✗' if err else '✓'} {_first_line(m.content)}",
                                          style="red" if err else "dim", markup=False)
        return None

    payload = {"messages": [HumanMessage(
        content=bug_report or f"Fix the bug reported at {issue_url}")]}
    while True:
        pause = asyncio.run(_drive(payload))
        if pause is None:
            break
        summary = pause.get("summary") if isinstance(pause, dict) else str(pause)
        console.print(Panel(str(summary), title="[yellow]Needs your input[/yellow]", expand=False))
        feedback = typer.prompt("Your guidance (or 'skip' to stop)")
        payload = Command(resume=feedback)

    snapshot = asyncio.run(graph.aget_state(config))
    _print_summary((snapshot.values or {}).get("debug"))


if __name__ == "__main__":
    app()
