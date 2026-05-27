"""CLI entry point for AutoDebug."""

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


@app.command()
def debug(
    repo_url: str = typer.Argument(..., help="GitHub repo URL to debug"),
    bug_report: str = typer.Option(..., "--bug", "-b", help="Bug report text"),
    issue_url: Optional[str] = typer.Option(None, "--issue", "-i", help="GitHub issue URL"),
    known_good: Optional[str] = typer.Option(None, "--good", "-g", help="Known good commit hash or date"),
):
    """Run the full AutoDebug pipeline on a repository."""
    console.print(Panel(f"[bold blue]AutoDebug[/bold blue]\n{repo_url}", expand=False))

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
