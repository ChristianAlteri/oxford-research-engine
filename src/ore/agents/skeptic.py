"""Skeptic agent — rigorous, adversarial critic."""

from ore.agents.base import BaseAgent
from ore.research_objects import SKEPTIC_TYPES


class SkepticAgent(BaseAgent):
    role = "skeptic"
    output_types = SKEPTIC_TYPES
    can_search = True
