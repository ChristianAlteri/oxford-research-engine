"""LLM provider layer — wraps LiteLLM with structured output parsing and cost tracking."""

from __future__ import annotations

import json
import logging
import re as _re
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from pydantic import BaseModel

litellm.modify_params = True

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 15

_JSON_RETRY_USER = (
    "Your previous reply had no extractable text (or it was empty). "
    "Output ONLY one JSON object matching the OUTPUT FORMAT in the system message. "
    "No markdown code fences, no preamble, no explanation outside JSON, no tool calls."
)


def _call_with_retry(fn: Any, **kwargs: Any) -> Any:
    """Call a litellm function with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate_limit" in err_str or "rate limit" in err_str
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            from rich.console import Console
            Console(stderr=True).print(
                f"[dim]⏳ Rate limited. Waiting {delay}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})...[/dim]"
            )
            time.sleep(delay)


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


def _extract_text_content(message: Any) -> str:
    """Pull text from a LiteLLM / Anthropic message, handling list-of-blocks format."""
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content

    # LiteLLM often sets content to a list of blocks: [{"type":"text","text":"..."}, ...]
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    # Anthropic raw on message object
    raw = getattr(message, "_raw_response", None)
    if raw and hasattr(raw, "content") and isinstance(raw.content, list):
        parts = [
            block.get("text", "")
            for block in raw.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)

    # model_dump() fallback
    try:
        dumped = message.model_dump() if hasattr(message, "model_dump") else {}
        raw_content = dumped.get("content", "")
        if isinstance(raw_content, list):
            parts = []
            for b in raw_content:
                if isinstance(b, dict):
                    if b.get("type") == "text" and "text" in b:
                        parts.append(str(b["text"]))
                    elif "text" in b:
                        parts.append(str(b["text"]))
            if parts:
                return "\n".join(parts)
        if isinstance(raw_content, str) and raw_content.strip():
            return raw_content
    except Exception:
        pass

    # Some providers put assistant text in reasoning_content (or similar)
    for attr in ("reasoning_content", "provider_specific_fields"):
        extra = getattr(message, attr, None)
        if isinstance(extra, str) and extra.strip():
            return extra

    return (content if isinstance(content, str) else "") or ""


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, handling various wrapper formats."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown fences
    if "```" in text:
        match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, _re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Find first { ... last } (greedy brace matching)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


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
        tools: list[dict] | None = None,
        tool_executor: Any | None = None,
    ) -> str:
        """Make an LLM completion call and return the raw text response.

        If output_types is provided, the system prompt is augmented with
        JSON schema instructions so the LLM returns structured output.

        If tools and tool_executor are provided, the LLM can call tools
        (like web search) mid-thought. The loop runs until the LLM produces
        a final text response or hits the max tool call limit.
        """
        full_system = system_prompt
        if output_types:
            schemas = _build_json_schema(output_types)
            type_names = [s["type_name"] for s in schemas]
            schema_block = json.dumps(schemas, indent=2)
            search_hint = ""
            if tools:
                search_hint = (
                    "Use web_search BEFORE your final JSON whenever you need real-world "
                    "facts, numbers, or study names. Do not invent statistics.\n"
                    "For Conjecture, ToyModel, and Reframe: fill `sources` as objects "
                    "{reference, credibility} per URL. credibility is 0.0–1.0 by source "
                    "type (peer review/gov/vendor primary ≈0.9+, news ≈0.6, Reddit/forums "
                    "≈0.2). If no web facts, `sources` may be [].\n"
                )
            full_system += (
                "\n\n--- OUTPUT FORMAT ---\n"
                f"You MUST respond with a JSON object matching ONE of these types: "
                f"{', '.join(type_names)}.\n\n"
                f"Schemas:\n{schema_block}\n\n"
                "Your response must be ONLY valid JSON. No markdown fences, no preamble, "
                "no explanation outside the JSON. Include an 'object_type' field set to "
                "the lowercase type name (e.g. 'conjecture', 'objection').\n"
                f"{search_hint}"
                "--- END OUTPUT FORMAT ---"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ]

        if tools and tool_executor:
            return self._complete_with_tools(
                model=model,
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
                temperature=temperature,
                max_tokens=max_tokens,
                round_number=round_number,
            )

        response = _call_with_retry(
            litellm.completion,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.usage.record(response, round_number=round_number)
        text = _extract_text_content(response.choices[0].message)
        if not text.strip() and output_types:
            text = self._retry_empty_json_output(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                round_number=round_number,
            )
        return text

    def _retry_empty_json_output(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        round_number: int,
    ) -> str:
        """One follow-up completion without tools when the model returned no text."""
        logger.warning("Empty model output; sending JSON-only retry.")
        from rich.console import Console

        Console(stderr=True).print(
            "[dim]⚠ Empty response — retrying with JSON-only instruction...[/dim]"
        )
        messages_retry = [*messages, {"role": "user", "content": _JSON_RETRY_USER}]
        response = _call_with_retry(
            litellm.completion,
            model=model,
            messages=messages_retry,
            temperature=min(temperature, 0.3),
            max_tokens=max_tokens,
        )
        self.usage.record(response, round_number=round_number)
        return _extract_text_content(response.choices[0].message)

    def _complete_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        tool_executor: Any,
        temperature: float,
        max_tokens: int,
        round_number: int,
        max_tool_rounds: int = 5,
    ) -> str:
        """Run a tool-calling loop: LLM can search, then produce final output."""
        for _ in range(max_tool_rounds):
            response = _call_with_retry(
                litellm.completion,
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.usage.record(response, round_number=round_number)
            choice = response.choices[0]

            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                text = _extract_text_content(choice.message)
                if not text.strip():
                    messages.append(choice.message.model_dump())
                    text = self._retry_empty_json_output(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        round_number=round_number,
                    )
                return text

            messages.append(choice.message.model_dump())

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                logger.info("Tool call: %s(%s)", fn_name, fn_args)

                result = tool_executor(
                    fn_name, fn_args, round_number=round_number
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # No tools on this call — force a text/JSON turn after tool budget exhausted
        response = _call_with_retry(
            litellm.completion,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.usage.record(response, round_number=round_number)
        final_msg = response.choices[0].message
        text = _extract_text_content(final_msg)
        if not text.strip():
            messages.append(final_msg.model_dump())
            text = self._retry_empty_json_output(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                round_number=round_number,
            )
        return text

    def parse_structured_output(
        self,
        raw: str,
        output_types: list[type[BaseModel]],
    ) -> BaseModel:
        """Parse raw LLM text into a typed Pydantic model.

        Aggressively extracts JSON from the response — handles markdown fences,
        preamble text, trailing text, and nested JSON.
        """
        text = raw.strip()
        data = _extract_json(text)

        type_map = {cls.__name__.lower(): cls for cls in output_types}
        obj_type = data.get("object_type", "").lower().replace("_", "")

        cls = type_map.get(obj_type)
        if cls is None:
            cls = output_types[0]

        return cls.model_validate(data)

    @property
    def budget_spent(self) -> float:
        return self.usage.total_cost_usd
