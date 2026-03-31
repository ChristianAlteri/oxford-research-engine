"""Typed research objects — the structured outputs of all agent interactions.

Every agent output is one of these types. No free-form prose allowed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ResearchObject(BaseModel):
    """Base class for all research objects produced by agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    object_type: str = ""
    round_number: int = 0
    created_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.object_type:
            self.object_type = type(self).__name__.lower()


# --- Builder outputs ---


class Conjecture(ResearchObject):
    """A proposed claim or hypothesis to be tested."""

    claim: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of prior research objects this builds on",
    )


class ToyModel(ResearchObject):
    """A simplified model that captures the essence of an idea."""

    description: str
    assumptions: list[str]
    predictions: list[str]
    limitations: list[str]


class Reframe(ResearchObject):
    """A new way of looking at the problem that shifts the debate."""

    original_framing: str
    new_framing: str
    justification: str


# --- Skeptic outputs ---


class Objection(ResearchObject):
    """A challenge to a specific research object."""

    target_id: str = Field(description="ID of the object being challenged")
    critique: str
    severity: Literal["minor", "major", "fatal"]
    suggested_fix: str | None = None


class FalsificationTest(ResearchObject):
    """A proposed test that could disprove a conjecture."""

    target_id: str = Field(description="ID of the conjecture to test")
    test_description: str
    expected_outcome_if_false: str
    feasibility: Literal["easy", "moderate", "hard", "theoretical"]


class CounterExample(ResearchObject):
    """A specific case that contradicts a claim."""

    target_id: str = Field(description="ID of the object being contradicted")
    example: str
    why_it_contradicts: str


# --- Historian outputs ---


class Synthesis(ResearchObject):
    """A summary of the current state of the research."""

    surviving_ideas: list[str]
    killed_ideas: list[str]
    open_questions: list[str]
    narrative: str


class PatternNote(ResearchObject):
    """An observation about recurring themes or dynamics."""

    pattern: str
    evidence: list[str]
    significance: str


class OpenQuestion(ResearchObject):
    """A question that remains unanswered and deserves attention."""

    question: str
    why_it_matters: str
    related_objects: list[str] = Field(default_factory=list)


# --- Referee outputs ---


class RoundScore(ResearchObject):
    """The Referee's assessment of a round."""

    novelty: float = Field(ge=0.0, le=10.0)
    rigor: float = Field(ge=0.0, le=10.0)
    convergence: float = Field(ge=0.0, le=10.0)
    verdict: Literal["continue", "stop", "pivot"]
    rationale: str


# --- Type registry ---

RESEARCH_OBJECT_TYPES: dict[str, type[ResearchObject]] = {
    "conjecture": Conjecture,
    "toymodel": ToyModel,
    "reframe": Reframe,
    "objection": Objection,
    "falsificationtest": FalsificationTest,
    "counterexample": CounterExample,
    "synthesis": Synthesis,
    "patternnote": PatternNote,
    "openquestion": OpenQuestion,
    "roundscore": RoundScore,
}

BUILDER_TYPES = [Conjecture, ToyModel, Reframe]
SKEPTIC_TYPES = [Objection, FalsificationTest, CounterExample]
HISTORIAN_TYPES = [Synthesis, PatternNote, OpenQuestion]
REFEREE_TYPES = [RoundScore]


def deserialize_research_object(data: dict) -> ResearchObject:
    """Reconstruct a typed ResearchObject from a dictionary."""
    obj_type = data.get("object_type", "")
    cls = RESEARCH_OBJECT_TYPES.get(obj_type, ResearchObject)
    return cls.model_validate(data)
