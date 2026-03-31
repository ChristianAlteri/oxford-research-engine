"""Export research sessions to markdown and JSON formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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


class MarkdownExporter:
    """Exports a research session to a readable markdown report."""

    def export(
        self,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
        hitl_entries: list[dict] | None = None,
    ) -> str:
        lines = [
            "# ORE Research Report",
            "",
            f"**Session**: `{session_id}`  ",
            f"**Question**: {question}  ",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
            "",
        ]

        if cost_summary:
            lines.extend([
                f"**Tokens used**: {cost_summary.get('total_tokens', 'N/A')}  ",
                f"**Estimated cost**: ${cost_summary.get('total_cost_usd', 0):.4f}  ",
                "",
            ])

        if hitl_entries:
            lines.extend([
                "## Suggested human review (rule-based)",
                "",
                "*Heuristic flags only — not model self-confidence. Review before citing.*",
                "",
            ])
            for e in hitl_entries:
                fl = ", ".join(e.get("flags", []))
                lines.append(
                    f"- **Round {e.get('round_number', '?')}** · `{e.get('object_id', '')}` "
                    f"({e.get('phase', '')}): {fl}"
                )
            lines.append("")

        lines.append("---\n")

        rounds = {}
        for obj in objects:
            rounds.setdefault(obj.round_number, []).append(obj)

        for round_num in sorted(rounds.keys()):
            round_objs = rounds[round_num]
            lines.append(f"## Round {round_num}\n")

            for obj in round_objs:
                lines.append(self._render_object(obj))
                lines.append("")

            scores = [o for o in round_objs if isinstance(o, RoundScore)]
            if scores:
                s = scores[-1]
                lines.extend([
                    f"### Round {round_num} Score",
                    "",
                    "| Metric | Score |",
                    "|--------|-------|",
                    f"| Novelty | {s.novelty}/10 |",
                    f"| Rigor | {s.rigor}/10 |",
                    f"| Convergence | {s.convergence}/10 |",
                    f"| **Verdict** | **{s.verdict.upper()}** |",
                    "",
                    f"> {s.rationale}",
                    "",
                ])

            lines.append("---\n")

        final_synth = [o for o in objects if isinstance(o, Synthesis)]
        if final_synth:
            last = final_synth[-1]
            lines.extend([
                "## Final Synthesis\n",
                f"{last.narrative}\n",
                "### Surviving Ideas",
                *[f"- {idea}" for idea in last.surviving_ideas],
                "",
                "### Killed Ideas",
                *[f"- ~~{idea}~~" for idea in last.killed_ideas],
                "",
                "### Open Questions",
                *[f"- {q}" for q in last.open_questions],
                "",
            ])

        return "\n".join(lines)

    def _render_object(self, obj: ResearchObject) -> str:
        label = obj.object_type.upper()
        header = f"#### {label} `{obj.id}` _{obj.created_by}_\n"

        if isinstance(obj, Conjecture):
            conf_bar = "█" * int(obj.confidence * 10) + "░" * (10 - int(obj.confidence * 10))
            src = ""
            if obj.sources:
                src = "\n\n**Sources** (credibility 0–1):\n" + "\n".join(
                    f"- [{s.credibility:.2f}] {s.reference}" for s in obj.sources
                )
            return (
                f"{header}"
                f"**Claim**: {obj.claim}\n\n"
                f"{obj.reasoning}\n\n"
                f"Confidence: [{conf_bar}] {obj.confidence:.0%}"
                f"{src}"
            )

        if isinstance(obj, ToyModel):
            assumptions = "\n".join(f"  - {a}" for a in obj.assumptions)
            predictions = "\n".join(f"  - {p}" for p in obj.predictions)
            limitations = "\n".join(f"  - {lim}" for lim in obj.limitations)
            src = ""
            if obj.sources:
                src = "\n\n**Sources** (credibility 0–1):\n" + "\n".join(
                    f"- [{s.credibility:.2f}] {s.reference}" for s in obj.sources
                )
            return (
                f"{header}"
                f"{obj.description}\n\n"
                f"**Assumptions**:\n{assumptions}\n\n"
                f"**Predictions**:\n{predictions}\n\n"
                f"**Limitations**:\n{limitations}"
                f"{src}"
            )

        if isinstance(obj, Reframe):
            src = ""
            if obj.sources:
                src = "\n\n**Sources** (credibility 0–1):\n" + "\n".join(
                    f"- [{s.credibility:.2f}] {s.reference}" for s in obj.sources
                )
            return (
                f"{header}"
                f"**From**: {obj.original_framing}\n\n"
                f"**To**: {obj.new_framing}\n\n"
                f"{obj.justification}"
                f"{src}"
            )

        if isinstance(obj, Objection):
            icon = {"minor": "⚠️", "major": "🔴", "fatal": "💀"}.get(obj.severity, "❓")
            return (
                f"{header}"
                f"{icon} **Severity**: {obj.severity} | **Target**: `{obj.target_id}`\n\n"
                f"{obj.critique}"
                + (f"\n\n**Suggested fix**: {obj.suggested_fix}" if obj.suggested_fix else "")
            )

        if isinstance(obj, FalsificationTest):
            return (
                f"{header}"
                f"**Target**: `{obj.target_id}` | **Feasibility**: {obj.feasibility}\n\n"
                f"{obj.test_description}\n\n"
                f"**If false**: {obj.expected_outcome_if_false}"
            )

        if isinstance(obj, CounterExample):
            return (
                f"{header}"
                f"**Target**: `{obj.target_id}`\n\n"
                f"{obj.example}\n\n"
                f"**Why it contradicts**: {obj.why_it_contradicts}"
            )

        if isinstance(obj, Synthesis):
            surviving = ", ".join(obj.surviving_ideas) if obj.surviving_ideas else "None yet"
            killed = ", ".join(obj.killed_ideas) if obj.killed_ideas else "None yet"
            return (
                f"{header}"
                f"{obj.narrative}\n\n"
                f"**Surviving**: {surviving}  \n"
                f"**Killed**: {killed}"
            )

        if isinstance(obj, PatternNote):
            evidence = ", ".join(obj.evidence)
            return (
                f"{header}"
                f"**Pattern**: {obj.pattern}\n\n"
                f"**Evidence**: {evidence}\n\n"
                f"{obj.significance}"
            )

        if isinstance(obj, OpenQuestion):
            return f"{header}" f"**{obj.question}**\n\n" f"{obj.why_it_matters}"

        if isinstance(obj, RoundScore):
            return ""

        exclude = {"id", "object_type", "created_by", "created_at", "round_number"}
        data = obj.model_dump(exclude=exclude)
        return f"{header}" + "\n".join(f"**{k}**: {v}" for k, v in data.items())

    def save(
        self,
        path: Path,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
        hitl_entries: list[dict] | None = None,
    ) -> None:
        md = self.export(objects, question, session_id, cost_summary, hitl_entries)
        path.write_text(md)


class JsonExporter:
    """Exports a research session as structured JSON."""

    def export(
        self,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
        hitl_entries: list[dict] | None = None,
    ) -> str:
        data = {
            "session_id": session_id,
            "question": question,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cost": cost_summary,
            "hitl_review": hitl_entries or [],
            "objects": [obj.model_dump(mode="json") for obj in objects],
        }
        return json.dumps(data, indent=2, default=str)

    def save(
        self,
        path: Path,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
        hitl_entries: list[dict] | None = None,
    ) -> None:
        path.write_text(self.export(objects, question, session_id, cost_summary, hitl_entries))
