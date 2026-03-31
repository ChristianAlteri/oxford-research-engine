"""Referee agent — impartial judge and scoring."""

from ore.agents.base import BaseAgent
from ore.research_objects import REFEREE_TYPES


class RefereeAgent(BaseAgent):
    role = "referee"
    output_types = REFEREE_TYPES
