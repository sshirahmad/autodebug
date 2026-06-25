"""CLI entry point for AutoDebug."""

import asyncio
import uuid
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()

from autodebug.telemetry import setup_tracing
from autodebug.graph import run_pipeline

app = typer.Typer(help="AutoDebug — reproduce, bisect, and fix bugs automatically")
console = Console()


def _run_streaming(repo_url, bug_report, issue_url, known_good, stream_tokens=False):
    """Drive the interactive graph and print progress live (instead of the blocking
    run_pipeline). On a HITL interrupt, prompt the developer at the terminal and
    resume with their feedback."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    from autodebug.graph import build_graph

    graph = build_graph()
    config = {"configurable": {
        "thread_id": uuid.uuid4().hex, "repo_url": repo_url,
        "issue_url": issue_url, "known_good": known_good,
        "stream_tokens": stream_tokens,
    }}

    async def _drive(payload):
        async for mode, chunk in graph.astream(
            payload, config=config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                if "token" in chunk:
                    console.print(chunk["token"], end="", soft_wrap=True)  # inline tokens
                else:
                    console.print(f"[dim]{chunk.get('progress', '')}[/dim]")
            elif "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                first = intr[0] if isinstance(intr, (list, tuple)) else intr
                return getattr(first, "value", first)        # awaiting feedback
            else:
                for update in chunk.values():
                    for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                        if getattr(m, "content", ""):
                            console.print(m.content)
        return None  # finished

    payload = {"messages": [HumanMessage(content=bug_report)]}
    while True:
        pause = asyncio.run(_drive(payload))
        if pause is None:
            break
        console.print(Panel(str(pause), title="[yellow]Needs your input[/yellow]", expand=False))
        feedback = typer.prompt("Your guidance (or 'skip' to give up)")
        if feedback.strip().lower() == "skip":
            break
        payload = Command(resume=feedback)


@app.command()
def debug(
    repo_url: str = typer.Argument(..., help="GitHub repo URL to debug"),
    bug_report: str = typer.Option(..., "--bug", "-b", help="Bug report text"),
    issue_url: Optional[str] = typer.Option(None, "--issue", "-i", help="GitHub issue URL"),
    known_good: Optional[str] = typer.Option(None, "--good", "-g", help="Known good commit hash or date"),
    stream: bool = typer.Option(False, "--stream", help="Show live progress and prompt for feedback if stuck"),
    tokens: bool = typer.Option(False, "--tokens", help="Stream sub-agent LLM tokens live (implies --stream)"),
):
    """Run the full AutoDebug pipeline on a repository."""
    console.print(Panel(f"[bold blue]AutoDebug[/bold blue]\n{repo_url}", expand=False))

    if stream or tokens:
        _run_streaming(repo_url, bug_report, issue_url, known_good, stream_tokens=tokens)
        return

    state = run_pipeline(
        repo_url=repo_url,
        bug_report=bug_report,
        github_issue_url=issue_url,
        known_good_commit=known_good,
    )

    if state.repro and state.repro.confirmed:
        console.print("[green]✓ Bug reproduced[/green]")
    if state.bisect:
        console.print(f"[green]✓ Culprit commit: {state.bisect.culprit_commit[:8]}[/green]")
        console.print(f"  {state.bisect.commit_message}")
    if state.root_cause:
        console.print(f"[green]✓ Root cause:[/green] {state.root_cause.summary}")
    if state.fix:
        console.print("[green]✓ Fix applied[/green]")
        if state.fix.pr_url:
            console.print(f"  PR: {state.fix.pr_url}")

    if state.error:
        console.print(f"[red]✗ Failed at stage '{state.stage}': {state.error}[/red]")

    console.print(
        f"\n[dim]LLM calls: {state.total_llm_calls} | "
        f"Tokens: {state.total_tokens:,}[/dim]"
    )


if __name__ == "__main__":
    app()
