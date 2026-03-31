"""Rule-based flags for selective human review (no AI self-scored confidence).

Implements the spirit of ORE's self-analysis: external, inspectable triggers —
quantitative claims, policy/medical/financial stakes, weak source credibility —
not propagated model confidence.
"""

from __future__ import annotations

import re
from typing import Any

from ore.research_objects import (
    Conjecture,
    Reframe,
    ResearchObject,
    SourceCitation,
    ToyModel,
)

# High-stakes domains: flag for human judgment (simple keyword heuristic)
_STAKES_RE = re.compile(
    r"(?i)\b("
    r"medical|clinical|diagnos|prescri|vaccine|patient|cancer|covid|"
    r"financial|investment|sec\b|tax\b|legal|lawsuit|court|regulation|"
    r"policy|legislat|election|national security|bioweapon|"
    r"recommend(?:s|ation)?\s+(that|we|you)|"
    r"should\s+(ban|mandate|require)"
    r")\b"
)

_QUANT_RE = re.compile(
    r"(?i)\d[\d,]*\s*%|\$[\d,]|\b\d+\s*(million|billion|trillion)\b|"
    r"\bcagr\b|p\s*[<=]=?\s*0\.\d+"
)


def _text_blob(obj: ResearchObject) -> str:
    """Flatten an object to searchable text."""
    data = obj.model_dump(mode="json")
    parts: list[str] = []
    for k, v in data.items():
        if k in ("id", "object_type", "created_at", "created_by", "round_number"):
            continue
        if isinstance(v, (list, dict)):
            parts.append(str(v))
        elif v is not None:
            parts.append(str(v))
    return "\n".join(parts)


def _mean_credibility(sources: list[Any]) -> float | None:
    if not sources:
        return None
    scores: list[float] = []
    for s in sources:
        if isinstance(s, SourceCitation):
            scores.append(s.credibility)
        elif isinstance(s, dict) and "credibility" in s:
            scores.append(float(s["credibility"]))
    if not scores:
        return None
    return sum(scores) / len(scores)


def evaluate_research_object(obj: ResearchObject) -> list[str]:
    """Return human-review flag codes for this object (may be empty)."""
    flags: list[str] = []
    text = _text_blob(obj)

    if _STAKES_RE.search(text):
        flags.append("stakes_domain")

    if _QUANT_RE.search(text):
        flags.append("quantitative_claim")

    if isinstance(obj, (Conjecture, ToyModel, Reframe)):
        sources = getattr(obj, "sources", None) or []
        mean_cred = _mean_credibility(sources)
        if mean_cred is not None and mean_cred < 0.45:
            flags.append("low_mean_source_credibility")
        if _QUANT_RE.search(text) and not sources:
            flags.append("numbers_without_sources")

    # Policy-style recommendations often need scrutiny
    if re.search(
        r"(?i)\b(must|should|need to|critical that|essential to)\s+",
        text,
    ) and ("stakes_domain" in flags or "quantitative_claim" in flags):
        flags.append("prescriptive_plus_stakes_or_numbers")

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
