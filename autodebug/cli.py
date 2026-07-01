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
    bug_report: str = typer.Option(..., "--bug", "-b", help="Bug report text"),
    issue_url: Optional[str] = typer.Option(None, "--issue", "-i", help="GitHub issue URL"),
    known_good: Optional[str] = typer.Option(None, "--good", "-g",
                                             help="Known good commit hash or date"),
    unattended: bool = typer.Option(
        False, "--unattended",
        help="Never pause for input (CI/batch). By default AutoDebug pauses to ask you "
             "when a stage gets stuck or the budget runs out."),
):
    """Reproduce, bisect, root-cause, and fix a bug — pausing to ask for guidance when
    stuck (use --unattended to disable prompts)."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    from autodebug.graph import build_graph

    console.print(Panel(f"[bold blue]AutoDebug[/bold blue]\n{repo_url}", expand=False))
    graph = build_graph(hitl=not unattended)
    config = {"configurable": {"thread_id": uuid.uuid4().hex, "repo_url": repo_url,
                               "issue_url": issue_url, "known_good": known_good}}

    async def _drive(payload):
        # subgraphs=True so the Manager subgraph's live output streams and its
        # interrupts surface; `messages` = token-level text, `updates` = interrupts.
        async for _ns, mode, chunk in graph.astream(
            payload, config=config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            if mode == "messages":
                msg, _meta = chunk
                if getattr(msg, "content", ""):
                    console.print(msg.content, end="", soft_wrap=True)
            elif mode == "updates" and "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                first = intr[0] if isinstance(intr, (list, tuple)) else intr
                return getattr(first, "value", first)        # paused for input
        return None

    payload = {"messages": [HumanMessage(content=bug_report)]}
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
