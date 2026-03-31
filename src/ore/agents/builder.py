"""Builder agent — generative, optimistic thinker."""

from ore.agents.base import BaseAgent
from ore.research_objects import BUILDER_TYPES


class BuilderAgent(BaseAgent):
    role = "builder"
    output_types = BUILDER_TYPES
