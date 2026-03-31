"""Rich terminal renderer — live display of the research loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ore.research_objects import (
    Conjecture,
    CounterExample,
    FalsificationTest,
    Objection,
    OpenQuestion,
    PatternNote,
    Reframe,
    ResearchObject,
    RoundScore,
    Synthesis,
    ToyModel,
)

if TYPE_CHECKING:
    from ore.llm.provider import UsageStats

ROLE_STYLES = {
    "builder": ("bold cyan", "🔨"),
    "skeptic": ("bold red", "🔍"),
    "historian": ("bold yellow", "📜"),
    "referee": ("bold green", "⚖️"),
}

PHASE_LABELS = {
    "builder_propose": "Builder proposes",
    "skeptic_challenge": "Skeptic challenges",
    "builder_respond": "Builder responds",
    "historian_synthesize": "Historian synthesizes",
    "referee_score": "Referee scores",
}


class TerminalRenderer:
    """Renders research objects to the terminal using Rich."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_header(self, question: str, config_summary: str = "") -> None:
        header = Text()
        header.append("ORE", style="bold magenta")
        header.append(" — Oxford Research Engine\n", style="dim")
        header.append(f"\n{question}", style="bold white")

        self.console.print(Panel(header, border_style="magenta", padding=(1, 2)))
        if config_summary:
            self.console.print(f"[dim]{config_summary}[/dim]\n")

    def print_round_start(self, round_number: int) -> None:
        self.console.rule(f"[bold magenta]Round {round_number}[/bold magenta]")
        self.console.print()

    def print_object(self, obj: ResearchObject, phase: str) -> None:
        role = obj.created_by
        style, icon = ROLE_STYLES.get(role, ("white", "•"))
        label = PHASE_LABELS.get(phase, phase)

        content = self._render_content(obj)
        panel = Panel(
            content,
            title=f"{icon} {label}",
            subtitle=f"[dim]{obj.object_type} {obj.id}[/dim]",
            border_style=style.replace("bold ", ""),
        )
        self.console.print(panel)
        self.console.print()

    def print_round_score(self, score: RoundScore) -> None:
        table = Table(show_header=True, header_style="bold green", expand=True)
        table.add_column("Metric", style="dim")
        table.add_column("Score", justify="center")
        table.add_column("Bar", justify="left")

        for metric, value in [
            ("Novelty", score.novelty),
            ("Rigor", score.rigor),
            ("Convergence", score.convergence),
        ]:
            bar = self._score_bar(value)
            table.add_row(metric, f"{value:.1f}", bar)

        verdict_style = {
            "continue": "green",
            "stop": "red",
            "pivot": "yellow",
        }.get(score.verdict, "white")

        table.add_row("Verdict", f"[bold {verdict_style}]{score.verdict.upper()}[/]", "")

        self.console.print(Panel(table, title="⚖️ Round Score", border_style="green"))
        self.console.print(f"  [dim]{score.rationale}[/dim]\n")

    def print_budget_warning(self, spent: float, budget: float) -> None:
        pct = (spent / budget) * 100
        self.console.print(
            f"[bold yellow]⚠ Budget: ${spent:.4f} / ${budget:.2f} "
            f"({pct:.0f}%)[/bold yellow]\n"
        )

    def print_final_summary(self, usage: UsageStats, session_id: str) -> None:
        summary = usage.summary()
        table = Table(title="Session Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Session ID", session_id)
        table.add_row("Total tokens", f"{summary['total_tokens']:,}")
        table.add_row("Prompt tokens", f"{summary['prompt_tokens']:,}")
        table.add_row("Completion tokens", f"{summary['completion_tokens']:,}")
        table.add_row("LLM calls", str(summary["call_count"]))
        table.add_row("Estimated cost", f"${summary['total_cost_usd']:.4f}")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def print_stop_reason(self, reason: str) -> None:
        self.console.print(f"\n[bold magenta]Research complete:[/bold magenta] {reason}\n")

    def _render_content(self, obj: ResearchObject) -> str:
        if isinstance(obj, Conjecture):
            conf = f"{obj.confidence:.0%}"
            return f"[bold]Claim:[/bold] {obj.claim}\n\n{obj.reasoning}\n\nConfidence: {conf}"

        if isinstance(obj, ToyModel):
            parts = [obj.description, "\n[bold]Assumptions:[/bold]"]
            parts.extend(f"  • {a}" for a in obj.assumptions)
            parts.append("\n[bold]Predictions:[/bold]")
            parts.extend(f"  • {p}" for p in obj.predictions)
            parts.append("\n[bold]Limitations:[/bold]")
            parts.extend(f"  • {lim}" for lim in obj.limitations)
            return "\n".join(parts)

        if isinstance(obj, Reframe):
            return (
                f"[bold]From:[/bold] {obj.original_framing}\n\n"
                f"[bold]To:[/bold] {obj.new_framing}\n\n"
                f"{obj.justification}"
            )

        if isinstance(obj, Objection):
            sev_map = {
                "minor": "[yellow]MINOR[/]",
                "major": "[red]MAJOR[/]",
                "fatal": "[bold red]FATAL[/]",
            }
            sev = sev_map.get(obj.severity, obj.severity)
            fix = ""
            if obj.suggested_fix:
                fix = f"\n\n[bold]Suggested fix:[/bold] {obj.suggested_fix}"
            return f"{sev} → target {obj.target_id}\n\n{obj.critique}{fix}"

        if isinstance(obj, FalsificationTest):
            return (
                f"Target: {obj.target_id} | Feasibility: {obj.feasibility}\n\n"
                f"{obj.test_description}\n\n"
                f"[bold]If false:[/bold] {obj.expected_outcome_if_false}"
            )

        if isinstance(obj, CounterExample):
            return (
                f"Target: {obj.target_id}\n\n"
                f"{obj.example}\n\n"
                f"[bold]Contradiction:[/bold] {obj.why_it_contradicts}"
            )

        if isinstance(obj, Synthesis):
            parts = [obj.narrative, ""]
            if obj.surviving_ideas:
                parts.append("[bold green]Surviving:[/bold green]")
                parts.extend(f"  ✓ {i}" for i in obj.surviving_ideas)
            if obj.killed_ideas:
                parts.append("[bold red]Killed:[/bold red]")
                parts.extend(f"  ✗ {i}" for i in obj.killed_ideas)
            if obj.open_questions:
                parts.append("[bold yellow]Open questions:[/bold yellow]")
                parts.extend(f"  ? {q}" for q in obj.open_questions)
            return "\n".join(parts)

        if isinstance(obj, PatternNote):
            return (
                f"[bold]Pattern:[/bold] {obj.pattern}\n\n"
                f"Evidence: {', '.join(obj.evidence)}\n\n"
                f"{obj.significance}"
            )

        if isinstance(obj, OpenQuestion):
            return f"[bold]{obj.question}[/bold]\n\n{obj.why_it_matters}"

        if isinstance(obj, RoundScore):
            return ""

        exclude = {"id", "object_type", "created_by", "created_at", "round_number"}
        data = obj.model_dump(exclude=exclude)
        return "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in data.items())

    @staticmethod
    def _score_bar(value: float, width: int = 20) -> str:
        filled = int((value / 10) * width)
        empty = width - filled
        return f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim] {value:.1f}/10"
