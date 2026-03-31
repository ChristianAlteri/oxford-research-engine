"""Base agent class — defines the contract all research agents follow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ore.config import AgentConfig
from ore.llm.provider import LLMProvider
from ore.research_objects import ResearchObject
from ore.tools.search import SEARCH_TOOL_SCHEMA, WebSearchTool

if TYPE_CHECKING:
    from ore.memory.store import MemoryStore

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class BaseAgent:
    """Abstract base for all research agents.

    Each agent has a role name, a system prompt loaded from a text file,
    a set of allowed output types, and access to the shared LLM provider.
    Optionally has web search access for evidence-based research.
    """

    role: str = "base"
    output_types: list[type[BaseModel]] = []
    can_search: bool = False

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMProvider,
        search_tool: WebSearchTool | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.search_tool = search_tool
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

        tools = None
        tool_executor = None
        if self.can_search and self.search_tool and self.search_tool.available:
            tools = [SEARCH_TOOL_SCHEMA]
            tool_executor = self.search_tool.execute_tool_call

        raw = self.llm.complete(
            model=self.config.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_types=self.output_types,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            round_number=round_number,
            tools=tools,
            tool_executor=tool_executor,
        )

        try:
            result = self.llm.parse_structured_output(raw, self.output_types)
        except Exception as e:
            logger.warning(
                "Failed to parse structured output from %s: %s", self.role, e
            )
            logger.warning("Raw output (first 500 chars): %.500s", raw)
            fallback_cls = self.output_types[0]
            result = self._fallback_parse(raw, fallback_cls)

        if isinstance(result, ResearchObject):
            result.created_by = self.role
            result.round_number = round_number

        return result

    def _fallback_parse(self, raw: str, cls: type[BaseModel]) -> ResearchObject:
        """Last-resort parse: fill required fields with raw text or defaults."""
        if not raw.strip():
            raw = (
                "[Empty or unparseable model response — possible refusal, "
                "rate limit, or tool/JSON formatting bug]"
            )
        fields = cls.model_fields
        data: dict = {"object_type": cls.__name__.lower(), "created_by": self.role}
        for name, info in fields.items():
            has_default = (
                info.default is not PydanticUndefined or info.default_factory
            )
            if name in data or has_default:
                continue
            anno = info.annotation
            if anno is str:
                data[name] = raw
            elif anno is float:
                data[name] = 0.5
            elif anno is int:
                data[name] = 0
            elif str(anno).startswith("list"):
                data[name] = []
            elif str(anno).startswith("typing.Literal"):
                args = getattr(anno, "__args__", None)
                if args:
                    data[name] = args[0]
        return cls.model_validate(data)
