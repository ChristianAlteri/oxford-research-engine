"""Base agent class — defines the contract all research agents follow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ore.config import AgentConfig
from ore.llm.provider import LLMProvider
from ore.research_objects import ResearchObject

if TYPE_CHECKING:
    from ore.memory.store import MemoryStore

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class BaseAgent:
    """Abstract base for all research agents.

    Each agent has a role name, a system prompt loaded from a text file,
    a set of allowed output types, and access to the shared LLM provider.
    """

    role: str = "base"
    output_types: list[type[BaseModel]] = []

    def __init__(self, config: AgentConfig, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            prompt_file = PROMPTS_DIR / f"{self.role}.txt"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text()
            else:
                self._system_prompt = f"You are the {self.role} agent."
        return self._system_prompt

    def build_context(self, memory: MemoryStore, round_number: int, question: str) -> str:
        """Build the user-facing prompt with relevant context from memory."""
        recent = memory.get_recent(limit=20)
        context_parts = [f"RESEARCH QUESTION: {question}\n"]

        if recent:
            context_parts.append("RECENT RESEARCH OBJECTS:")
            for obj in recent:
                context_parts.append(self._format_object(obj))

        context_parts.append(f"\nCURRENT ROUND: {round_number}")
        return "\n".join(context_parts)

    def _format_object(self, obj: ResearchObject) -> str:
        """Format a research object for context injection."""
        data = obj.model_dump(exclude={"created_at"})
        lines = [f"  [{obj.object_type.upper()}] (id={obj.id}, by={obj.created_by})"]
        for key, value in data.items():
            if key in ("id", "object_type", "created_by", "round_number"):
                continue
            if isinstance(value, list):
                lines.append(f"    {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)

    def think(
        self,
        memory: MemoryStore,
        round_number: int,
        question: str,
        extra_context: str = "",
    ) -> ResearchObject:
        """Run the agent: build context, call LLM, parse structured output."""
        user_prompt = self.build_context(memory, round_number, question)
        if extra_context:
            user_prompt += f"\n\n{extra_context}"

        raw = self.llm.complete(
            model=self.config.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_types=self.output_types,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            round_number=round_number,
        )

        try:
            result = self.llm.parse_structured_output(raw, self.output_types)
        except Exception as e:
            logger.warning("Failed to parse structured output from %s: %s", self.role, e)
            logger.debug("Raw output: %s", raw)
            fallback_cls = self.output_types[0]
            result = self._fallback_parse(raw, fallback_cls)

        if isinstance(result, ResearchObject):
            result.created_by = self.role
            result.round_number = round_number

        return result

    def _fallback_parse(self, raw: str, cls: type[BaseModel]) -> ResearchObject:
        """Last-resort parse: stuff the raw text into the first string field."""
        fields = cls.model_fields
        data: dict = {"object_type": cls.__name__.lower(), "created_by": self.role}
        for name, info in fields.items():
            if info.annotation is str:
                data[name] = raw
                break
        return cls.model_validate(data)
