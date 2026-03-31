"""Configuration models for ORE sessions."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096


class EngineConfig(BaseModel):
    """Full configuration for a research session."""

    question: str = ""
    max_rounds: int = 10
    budget_usd: float = 10.0

    agents: dict[str, AgentConfig] = Field(default_factory=lambda: {
        "builder": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.8),
        "skeptic": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.3),
        "historian": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.5),
        "referee": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.1),
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        agents_raw = raw.pop("agents", {})
        agents = {name: AgentConfig(**cfg) for name, cfg in agents_raw.items()}

        config = cls(**raw)
        if agents:
            config.agents.update(agents)
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
