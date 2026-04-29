"""CLI entry point for ORE — Oxford Research Engine."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()
from rich.console import Console
from rich.table import Table

from ore.config import EngineConfig
from ore.engine import ResearchEngine
from ore.llm.provider import LLMProvider
from ore.memory.store import JsonMemoryStore
from ore.output.export import JsonExporter, MarkdownExporter, TldrExporter
from ore.output.terminal import TerminalRenderer
from ore.tools.search import WebSearchTool

console = Console()
renderer = TerminalRenderer(console)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """ORE — Oxford Research Engine.

    Orchestrate LLM agents in structured research loops.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")


@cli.command()
@click.argument("question", required=False)
@click.option(
    "--config", "-c", "config_path",
    type=click.Path(exists=True), help="YAML config file.",
)
@click.option("--rounds", "-r", type=int, help="Max rounds (overrides config).")
@click.option("--budget", "-b", type=float, help="Budget in USD (overrides config).")
@click.option("--session-id", "-s", type=str, help="Custom session ID.")
@click.option(
    "--export-format",
    "-f",
    type=click.Choice(["markdown", "json", "both", "tldr"]),
    default="markdown",
    help="Export format for the report.",
)
@click.option(
    "--delay", "-d", type=float, default=12,
    help="Seconds to wait between agent calls (rate-limit pacing). Default 12.",
)
@click.option(
    "--hitl/--no-hitl",
    default=True,
    help="Rule-based human-review flags + hitl_review.json (default: on).",
)
@click.option(
    "--pause",
    "hitl_pause",
    is_flag=True,
    help="Wait for Enter after each full round (human checkpoint).",
)
def run(
    question: str | None,
    config_path: str | None,
    rounds: int | None,
    budget: float | None,
    session_id: str | None,
    export_format: str,
    delay: float,
    hitl: bool,
    hitl_pause: bool,
) -> None:
    """Run a research session on a question."""
    if config_path:
        config = EngineConfig.from_yaml(config_path)
    else:
        config = EngineConfig()

    if question:
        config.question = question
    if rounds:
        config.max_rounds = rounds
    if budget:
        config.budget_usd = budget
    config.hitl = hitl
    config.hitl_pause = hitl_pause

    if not config.question:
        console.print("[bold red]Error:[/bold red] No question provided.")
        console.print("Usage: ore run \"Your research question here\"")
        sys.exit(1)

    memory = JsonMemoryStore(session_id=session_id, question=config.question)
    memory.ensure_dir()
    memory.save_config(config.model_dump())

    llm = LLMProvider()

    search_tool = WebSearchTool()
    search_enabled = search_tool.available

    engine = ResearchEngine(
        config=config,
        memory=memory,
        llm=llm,
        on_object=renderer.print_object,
        on_round_start=renderer.print_round_start,
        on_round_end=lambda r, s: renderer.print_round_score(s),
        on_budget_warning=renderer.print_budget_warning,
        on_hitl_flags=renderer.print_hitl_flags,
        search_tool=search_tool if search_enabled else None,
        call_delay=delay,
    )

    agent_models = ", ".join(f"{k}={v.model}" for k, v in config.agents.items())
    search_status = "on" if search_enabled else "off"
    delay_info = f" | Pacing: {delay:.0f}s" if delay > 0 else ""
    hitl_info = f" | HITL flags: {'on' if hitl else 'off'}"
    pause_info = " | Round pause: on" if hitl_pause else ""
    renderer.print_header(
        config.question,
        f"Max rounds: {config.max_rounds} | Budget: ${config.budget_usd:.2f} "
        f"| Search: {search_status}{delay_info}{hitl_info}{pause_info} | Models: {agent_models}",
    )
    if not search_enabled:
        console.print(
            "[dim]Tip: Set TAVILY_API_KEY to enable web search "
            "for Builder and Skeptic agents.[/dim]\n"
        )

    try:
        objects = engine.run()
    except KeyboardInterrupt:
        objects = memory.get_all()
        renderer.print_stop_reason("Interrupted by user.")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        sys.exit(1)

    _export_report(memory, objects, config, llm, export_format)

    renderer.print_final_summary(llm.usage, memory.session_id)
    console.print(f"[dim]Session saved: {memory.session_dir}[/dim]")


@cli.command()
@click.argument("session_id")
@click.option("--rounds", "-r", type=int, help="Additional rounds to run.")
@click.option("--budget", "-b", type=float, help="Additional budget in USD.")
@click.option("--hitl/--no-hitl", default=True, help="Rule-based HITL flags (default: on).")
@click.option(
    "--pause",
    "hitl_pause",
    is_flag=True,
    help="Wait for Enter after each full round.",
)
def resume(
    session_id: str,
    rounds: int | None,
    budget: float | None,
    hitl: bool,
    hitl_pause: bool,
) -> None:
    """Resume a previously stopped research session."""
    memory = JsonMemoryStore(session_id=session_id)
    if not memory.memory_file.exists():
        console.print(f"[bold red]Error:[/bold red] Session '{session_id}' not found.")
        sys.exit(1)

    memory.load()

    config_file = memory.session_dir / "config.yaml"
    if config_file.exists():
        config = EngineConfig.from_yaml(config_file)
    else:
        console.print("[bold red]Error:[/bold red] No config found for this session.")
        sys.exit(1)

    if rounds:
        current_max = max((o.round_number for o in memory.get_all()), default=0)
        config.max_rounds = current_max + rounds
    if budget:
        config.budget_usd = budget
    config.hitl = hitl
    config.hitl_pause = hitl_pause

    llm = LLMProvider()
    search_tool = WebSearchTool()
    search_enabled = search_tool.available

    engine = ResearchEngine(
        config=config,
        memory=memory,
        llm=llm,
        on_object=renderer.print_object,
        on_round_start=renderer.print_round_start,
        on_round_end=lambda r, s: renderer.print_round_score(s),
        on_budget_warning=renderer.print_budget_warning,
        on_hitl_flags=renderer.print_hitl_flags,
        search_tool=search_tool if search_enabled else None,
        call_delay=12,
    )

    renderer.print_header(config.question, f"Resuming session {session_id}")

    existing_rounds = max((o.round_number for o in memory.get_all()), default=0)
    console.print(f"[dim]Resuming from round {existing_rounds + 1}[/dim]\n")

    try:
        objects = engine.run()
    except KeyboardInterrupt:
        objects = memory.get_all()
        renderer.print_stop_reason("Interrupted by user.")

    _export_report(memory, objects, config, llm, "markdown")
    renderer.print_final_summary(llm.usage, memory.session_id)


@cli.command("list")
def list_sessions() -> None:
    """List all saved research sessions."""
    sessions = JsonMemoryStore.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions", show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="cyan")
    table.add_column("Question", max_width=50)
    table.add_column("Rounds", justify="center")
    table.add_column("Objects", justify="center")
    table.add_column("Verdict", justify="center")
    table.add_column("Modified", style="dim")

    for s in sessions:
        verdict = s.get("last_verdict") or "—"
        verdict_style = {"continue": "green", "stop": "red", "pivot": "yellow"}.get(verdict, "")
        modified = s["modified"].strftime("%Y-%m-%d %H:%M") if s.get("modified") else "—"

        table.add_row(
            s["session_id"],
            s.get("question", "—")[:50],
            str(s.get("rounds", 0)),
            str(s.get("total_objects", 0)),
            f"[{verdict_style}]{verdict}[/{verdict_style}]" if verdict_style else verdict,
            modified,
        )

    console.print(table)


@cli.command()
@click.argument("session_id")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "json", "both", "tldr"]),
    default="markdown",
    help="Export format. 'tldr' produces a shareable one-pager summary.",
)
@click.option("--output", "-o", "output_path", type=click.Path(), help="Output file path.")
def export(session_id: str, fmt: str, output_path: str | None) -> None:
    """Export a research session to markdown, JSON, or a TL;DR summary."""
    memory = JsonMemoryStore(session_id=session_id)
    if not memory.memory_file.exists():
        console.print(f"[bold red]Error:[/bold red] Session '{session_id}' not found.")
        sys.exit(1)

    memory.load()
    objects = memory.get_all()
    meta = memory.get_session_metadata()
    question = meta.get("question", "Unknown")

    hitl_entries = memory.get_hitl_entries()

    if fmt == "tldr":
        if output_path:
            tldr_path = Path(output_path)
        else:
            tldr_path = memory.session_dir / "tldr.md"
        TldrExporter().save(tldr_path, objects, question, session_id)
        console.print(f"[green]✓[/green] TL;DR summary: {tldr_path}")
        return

    if fmt in ("markdown", "both"):
        if output_path and fmt == "markdown":
            md_path = Path(output_path)
        else:
            md_path = memory.session_dir / "report.md"
        MarkdownExporter().save(
            md_path, objects, question, session_id, None, hitl_entries
        )
        console.print(f"[green]✓[/green] Markdown report: {md_path}")

    if fmt in ("json", "both"):
        if output_path and fmt == "json":
            json_path = Path(output_path)
        else:
            json_path = memory.session_dir / "report.json"
        JsonExporter().save(
            json_path, objects, question, session_id, None, hitl_entries
        )
        console.print(f"[green]✓[/green] JSON report: {json_path}")


def _export_report(
    memory: JsonMemoryStore,
    objects: list,
    config: EngineConfig,
    llm: LLMProvider,
    export_format: str,
) -> None:
    """Export the research report after a run."""
    cost = llm.usage.summary()
    hitl_entries = memory.get_hitl_entries()

    if export_format in ("markdown", "both"):
        md_path = memory.session_dir / "report.md"
        MarkdownExporter().save(
            md_path, objects, config.question, memory.session_id, cost, hitl_entries
        )

    if export_format in ("json", "both"):
        json_path = memory.session_dir / "report.json"
        JsonExporter().save(
            json_path, objects, config.question, memory.session_id, cost, hitl_entries
        )

    if export_format == "tldr":
        tldr_path = memory.session_dir / "tldr.md"
        TldrExporter().save(tldr_path, objects, config.question, memory.session_id, cost)


if __name__ == "__main__":
    cli()
