"""LLM provider layer — wraps LiteLLM with structured output parsing and cost tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm
from pydantic import BaseModel

litellm.drop_params = True

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Tracks cumulative token usage and estimated cost across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    per_round: dict[int, dict[str, Any]] = field(default_factory=dict)

    def record(self, response: Any, round_number: int = 0) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.call_count += 1

        try:
            cost = litellm.completion_cost(completion_response=response)
            self.total_cost_usd += cost
        except Exception:
            cost = 0.0

        if round_number not in self.per_round:
            self.per_round[round_number] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            }
        rd = self.per_round[round_number]
        rd["prompt_tokens"] += prompt
        rd["completion_tokens"] += completion
        rd["cost_usd"] += cost
        rd["calls"] += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
        }


def _build_json_schema(output_types: list[type[BaseModel]]) -> dict[str, Any]:
    """Build a JSON schema that accepts any of the given Pydantic model types.

    Returns a schema suitable for injecting into a system prompt to instruct
    the LLM to respond with valid JSON matching one of these types.
    """
    schemas = []
    for cls in output_types:
        schema = cls.model_json_schema()
        schema.pop("$defs", None)
        schemas.append({"type_name": cls.__name__, "schema": schema})
    return schemas


class LLMProvider:
    """Wraps LiteLLM to provide structured LLM calls with cost tracking."""

    def __init__(self) -> None:
        self.usage = UsageStats()

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_types: list[type[BaseModel]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        round_number: int = 0,
    ) -> str:
        """Make an LLM completion call and return the raw text response.

        If output_types is provided, the system prompt is augmented with
        JSON schema instructions so the LLM returns structured output.
        """
        full_system = system_prompt
        if output_types:
            schemas = _build_json_schema(output_types)
            type_names = [s["type_name"] for s in schemas]
            schema_block = json.dumps(schemas, indent=2)
            full_system += (
                "\n\n--- OUTPUT FORMAT ---\n"
                f"You MUST respond with a JSON object matching ONE of these types: "
                f"{', '.join(type_names)}.\n\n"
                f"Schemas:\n{schema_block}\n\n"
                "Your response must be ONLY valid JSON. No markdown fences, no preamble, "
                "no explanation outside the JSON. Include an 'object_type' field set to "
                "the lowercase type name (e.g. 'conjecture', 'objection').\n"
                "--- END OUTPUT FORMAT ---"
            )

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.usage.record(response, round_number=round_number)
        return response.choices[0].message.content or ""

    def parse_structured_output(
        self,
        raw: str,
        output_types: list[type[BaseModel]],
    ) -> BaseModel:
        """Parse raw LLM text into a typed Pydantic model.

        Tries to extract JSON from the response, then matches it to
        one of the provided output types based on the 'object_type' field.
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)

        type_map = {cls.__name__.lower(): cls for cls in output_types}
        obj_type = data.get("object_type", "").lower().replace("_", "")

        cls = type_map.get(obj_type)
        if cls is None:
            cls = output_types[0]

        return cls.model_validate(data)

    @property
    def budget_spent(self) -> float:
        return self.usage.total_cost_usd
