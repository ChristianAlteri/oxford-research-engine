"""Web search tool — gives agents access to real-time information."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class SearchStats:
    """Tracks search usage across a session."""

    total_searches: int = 0
    per_round: dict[int, int] = field(default_factory=dict)

    def record(self, round_number: int) -> None:
        self.total_searches += 1
        self.per_round[round_number] = self.per_round.get(round_number, 0) + 1


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current data, market statistics, peer-reviewed studies, "
            "competitor analysis, pricing data, or any factual claim that needs validation. "
            "Use this to ground your arguments in real evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific search query. Be precise "
                        "— include numbers, names, dates."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


class WebSearchTool:
    """Web search via Tavily API. Designed for AI agent research."""

    def __init__(self, max_searches_per_call: int = 3) -> None:
        self.max_searches_per_call = max_searches_per_call
        self.stats = SearchStats()
        self._client = None

    @property
    def available(self) -> bool:
        return bool(os.environ.get("TAVILY_API_KEY"))

    @property
    def client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            except ImportError:
                raise RuntimeError(
                    "tavily-python is required for web search. "
                    "Install it with: pip install tavily-python"
                )
        return self._client

    def search(self, query: str, round_number: int = 0) -> list[SearchResult]:
        """Execute a web search and return structured results."""
        self.stats.record(round_number)
        logger.info("Web search: %s", query)

        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query, e)
            return []

        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
            ))

        return results

    def format_results(self, results: list[SearchResult]) -> str:
        """Format search results for injection into LLM context."""
        if not results:
            return "No results found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.content}")
        return "\n\n".join(parts)

    def execute_tool_call(
        self, function_name: str, arguments: str | dict, round_number: int = 0
    ) -> str:
        """Execute a tool call from the LLM and return formatted results."""
        if function_name != "web_search":
            return f"Unknown tool: {function_name}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"query": arguments}

        query = arguments.get("query", "")
        if not query:
            return "No query provided."

        results = self.search(query, round_number=round_number)
        return self.format_results(results)
