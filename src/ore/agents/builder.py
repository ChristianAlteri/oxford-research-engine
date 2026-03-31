"""Builder agent — generative, optimistic thinker."""

from __future__ import annotations

import logging
import re

from ore.agents.base import BaseAgent
from ore.research_objects import BUILDER_TYPES, Conjecture, Reframe, ToyModel

logger = logging.getLogger(__name__)

# Triggers likely needing web_search + `sources` when search is enabled
_QUANT_OR_CITATION = re.compile(
    r"(?i)\d[\d,]*\s*%|\$[\d,]|\b\d{4}\b.*\b(study|report|survey)\b|"
    r"\b(cagr|billion|million)\b|% of|hours per|research shows|according to|"
    r"peer-reviewed|meta-analysis|doi:|issn:"
)


class BuilderAgent(BaseAgent):
    role = "builder"
    output_types = BUILDER_TYPES
    can_search = True

    def think(
        self,
        memory,
        round_number: int,
        question: str,
        extra_context: str = "",
    ):
        result = super().think(
            memory, round_number, question, extra_context=extra_context
        )
        if self.search_tool and self.search_tool.available:
            result = self._flag_ungrounded_quantitative(result)
        return result

    def _flag_ungrounded_quantitative(self, result):
        """Append a visible disclaimer when the model likely hallucinated facts."""
        if not isinstance(result, (Conjecture, ToyModel, Reframe)):
            return result
        text = self._combined_text(result)
        if not _QUANT_OR_CITATION.search(text):
            return result
        if getattr(result, "sources", None):
            return result  # has at least one SourceCitation
        logger.warning(
            "Builder used quantitative or citation-like language without `sources`."
        )
        note = (
            "\n\n[Grounding: No URLs in `sources`. Treat numbers and named studies "
            "above as unverified unless checked.]"
        )
        if isinstance(result, Conjecture):
            result.reasoning += note
        elif isinstance(result, ToyModel):
            result.limitations.append(
                "Grounding: no URLs in sources; verify any empirical claims."
            )
        else:
            result.justification += note
        return result

    @staticmethod
    def _combined_text(obj: Conjecture | ToyModel | Reframe) -> str:
        if isinstance(obj, Conjecture):
            return f"{obj.claim}\n{obj.reasoning}"
        if isinstance(obj, ToyModel):
            return "\n".join(
                [obj.description, *obj.assumptions, *obj.predictions]
            )
        return f"{obj.original_framing}\n{obj.new_framing}\n{obj.justification}"
