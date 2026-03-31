"""Historian agent — observant synthesizer."""

from ore.agents.base import BaseAgent
from ore.research_objects import HISTORIAN_TYPES


class HistorianAgent(BaseAgent):
    role = "historian"
    output_types = HISTORIAN_TYPES
