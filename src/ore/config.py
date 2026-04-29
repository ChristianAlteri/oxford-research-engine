"""Configuration models for ORE sessions."""

from __future__ import annotations

import html
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _html_to_plain_text(html_content: str) -> str:
    """Strip tags and scripts; keep readable paragraph breaks (no external deps)."""
    text = html_content
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", "", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
    # Rule-based human-in-the-loop hints (no model self-confidence)
    hitl: bool = True
    hitl_pause: bool = False

    agents: dict[str, AgentConfig] = Field(default_factory=lambda: {
        "builder": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.65),
        "skeptic": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.3),
        "historian": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.5),
        "referee": AgentConfig(model="anthropic/claude-sonnet-4-20250514", temperature=0.1),
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file."""
        yaml_path = Path(path).resolve()
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        source_document = raw.pop("source_document", None)

        agents_raw = raw.pop("agents", {})
        agents = {name: AgentConfig(**cfg) for name, cfg in agents_raw.items()}

        config = cls(**raw)
        if agents:
            config.agents.update(agents)

        if source_document:
            doc_path = Path(source_document).expanduser()
            if not doc_path.is_absolute():
                doc_path = yaml_path.parent / doc_path
            doc_path = doc_path.resolve()
            if doc_path.is_file():
                raw_html = doc_path.read_text(encoding="utf-8")
                plain = _html_to_plain_text(raw_html)
                sep = (
                    f"\n\n--- Source document ({doc_path}) — plain text ---\n\n"
                )
                base = (config.question or "").strip()
                config.question = base + sep + plain
            else:
                config.question = (config.question or "").strip() + (
                    f"\n\n[ORE config error: source_document not found: {doc_path}]"
                )

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
