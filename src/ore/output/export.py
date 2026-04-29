"""Export research sessions to markdown, JSON, and TL;DR formats."""

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


class TldrExporter:
    """Produces a shareable one-pager TL;DR from a research session.

    Format:
    - Question / framing
    - Ideas explored table (idea | what it gets right | what kills it)
    - Surviving ideas
    - Killed ideas
    - Open questions
    - One-line strategy summary (final Referee rationale)
    """

    def export(
        self,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
    ) -> str:
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        short_q = question.split("\n")[0].strip()[:200]
        lines += [
            f"# TL;DR — {short_q}",
            "",
            f"*Session `{session_id}` · "
            f"{datetime.now(timezone.utc).strftime('%B %Y')}*",
            "",
            "---",
            "",
        ]

        # ── Ideas explored table ──────────────────────────────────────────────
        # Pair each main proposition (Conjecture / ToyModel / Reframe) with
        # the highest-severity Objection that targets it (if any).

        props: list[Conjecture | ToyModel | Reframe] = [
            o for o in objects
            if isinstance(o, (Conjecture, ToyModel, Reframe))
        ]
        objections: list[Objection] = [
            o for o in objects if isinstance(o, Objection)
        ]

        # Index objections by target_id → keep worst severity
        sev_rank = {"fatal": 3, "major": 2, "minor": 1}
        obj_by_target: dict[str, Objection] = {}
        for obj in objections:
            existing = obj_by_target.get(obj.target_id)
            if existing is None or sev_rank.get(obj.severity, 0) > sev_rank.get(existing.severity, 0):
                obj_by_target[obj.target_id] = obj

        if props:
            lines += [
                "## Ideas explored",
                "",
                "| Idea | What it gets right | What kills it |",
                "|---|---|---|",
            ]
            for prop in props:
                # Label
                if isinstance(prop, Conjecture):
                    label = prop.claim[:80] + ("…" if len(prop.claim) > 80 else "")
                    positive = _first_sentence(prop.reasoning)
                elif isinstance(prop, ToyModel):
                    label = prop.description[:80] + ("…" if len(prop.description) > 80 else "")
                    positive = prop.predictions[0] if prop.predictions else "—"
                else:  # Reframe
                    label = prop.new_framing[:80] + ("…" if len(prop.new_framing) > 80 else "")
                    positive = _first_sentence(prop.justification)

                # Find objection
                killer = obj_by_target.get(prop.id)
                if killer:
                    icon = {"fatal": "💀", "major": "🔴", "minor": "⚠️"}.get(killer.severity, "")
                    neg = f"{icon} {_first_sentence(killer.critique)}"
                else:
                    neg = "—"

                lines.append(
                    f"| {_escape_table(label)} "
                    f"| {_escape_table(positive)} "
                    f"| {_escape_table(neg)} |"
                )
            lines.append("")

        # ── Final Synthesis ───────────────────────────────────────────────────
        syntheses: list[Synthesis] = [o for o in objects if isinstance(o, Synthesis)]
        if syntheses:
            last = syntheses[-1]

            if last.surviving_ideas:
                lines += ["## What survived", ""]
                for idea in last.surviving_ideas:
                    lines.append(f"- {idea}")
                lines.append("")

            if last.killed_ideas:
                lines += ["## What was killed", ""]
                for idea in last.killed_ideas:
                    lines.append(f"- ~~{idea}~~")
                lines.append("")

            if last.open_questions:
                lines += ["## Open questions", ""]
                for q in last.open_questions:
                    lines.append(f"- {q}")
                lines.append("")

        # ── One-line summary (final Referee verdict) ──────────────────────────
        scores: list[RoundScore] = [o for o in objects if isinstance(o, RoundScore)]
        if scores:
            final_score = scores[-1]
            verdict_icon = {"stop": "🔴", "continue": "🟡", "pivot": "🔵"}.get(
                final_score.verdict, ""
            )
            lines += [
                "---",
                "",
                f"**Final verdict**: {verdict_icon} `{final_score.verdict.upper()}` "
                f"— {final_score.rationale}",
                "",
            ]

        # ── Footer ────────────────────────────────────────────────────────────
        cost_note = ""
        if cost_summary:
            cost_note = (
                f" · ${cost_summary.get('total_cost_usd', 0):.4f} "
                f"({cost_summary.get('total_tokens', '?')} tokens)"
            )
        lines += [
            f"*Generated by ORE (Oxford Research Engine){cost_note}. "
            "Quantitative claims flagged for human review before citing.*",
        ]

        return "\n".join(lines)

    def save(
        self,
        path: Path,
        objects: list[ResearchObject],
        question: str,
        session_id: str,
        cost_summary: dict | None = None,
    ) -> None:
        path.write_text(self.export(objects, question, session_id, cost_summary))


def _first_sentence(text: str) -> str:
    """Return the first sentence (up to 120 chars) of a block of text."""
    text = text.strip().replace("\n", " ")
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if 0 < idx < 120:
            return text[: idx + 1].strip()
    return text[:120].strip()


def _escape_table(text: str) -> str:
    """Escape pipe characters for Markdown table cells."""
    return text.replace("|", "\\|")


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
