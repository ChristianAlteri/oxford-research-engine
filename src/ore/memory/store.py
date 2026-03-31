"""Memory store — JSON-backed persistence for research objects and sessions."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ore.research_objects import ResearchObject, RoundScore, deserialize_research_object

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why", "if", "than",
    "we", "our", "you", "your", "i", "my", "me", "he", "she", "they", "them",
    "between", "toward", "towards", "into", "about", "through", "during", "before",
    "after", "above", "below", "specific", "particular", "whether", "while",
    "also", "just", "like", "look", "explore", "consider", "understand",
}


def generate_session_slug(question: str, max_words: int = 4) -> str:
    """Generate a human-readable session slug from a research question.

    Examples:
        "What resolves QM and GR?" -> "resolves-qm-gr-a3f2"
        "How should Storage convert this user?" -> "storage-convert-user-8b1c"
    """
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", question.lower())
    words = [w for w in clean.split() if w not in STOPWORDS and len(w) > 1]
    slug_words = words[:max_words]

    if not slug_words:
        slug_words = ["session"]

    short_hash = hashlib.sha256(question.encode()).hexdigest()[:4]
    return "-".join(slug_words) + "-" + short_hash


class MemoryStore(Protocol):
    """Protocol for research memory backends."""

    def add(self, obj: ResearchObject) -> None: ...
    def get_all(self) -> list[ResearchObject]: ...
    def get_recent(self, limit: int = 10) -> list[ResearchObject]: ...
    def get_by_round(self, round_number: int) -> list[ResearchObject]: ...
    def get_by_type(self, object_type: str) -> list[ResearchObject]: ...
    def get_by_id(self, obj_id: str) -> ResearchObject | None: ...
    def get_scores(self) -> list[RoundScore]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...


DEFAULT_SESSIONS_DIR = Path.home() / ".ore" / "sessions"


class JsonMemoryStore:
    """JSON file-backed memory store. Each session gets its own directory."""

    def __init__(
        self,
        session_id: str | None = None,
        sessions_dir: Path | None = None,
        question: str | None = None,
    ) -> None:
        if session_id:
            self.session_id = session_id
        elif question:
            self.session_id = generate_session_slug(question)
        else:
            self.session_id = uuid.uuid4().hex[:8]
        self.sessions_dir = sessions_dir or DEFAULT_SESSIONS_DIR
        self.session_dir = self.sessions_dir / self.session_id
        self._objects: list[ResearchObject] = []
        self._index: dict[str, ResearchObject] = {}
        self._hitl_entries: list[dict] = []

    @property
    def memory_file(self) -> Path:
        return self.session_dir / "memory.json"

    def ensure_dir(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def add(self, obj: ResearchObject) -> None:
        self._objects.append(obj)
        self._index[obj.id] = obj

    def record_hitl(
        self,
        *,
        object_id: str,
        object_type: str,
        round_number: int,
        phase: str,
        flags: list[str],
    ) -> None:
        """Append a rule-based human-review flag record (audit trail)."""
        self._hitl_entries.append({
            "object_id": object_id,
            "object_type": object_type,
            "round_number": round_number,
            "phase": phase,
            "flags": flags,
        })

    def get_hitl_entries(self) -> list[dict]:
        return list(self._hitl_entries)

    def get_all(self) -> list[ResearchObject]:
        return list(self._objects)

    def get_recent(self, limit: int = 10) -> list[ResearchObject]:
        return self._objects[-limit:]

    def get_by_round(self, round_number: int) -> list[ResearchObject]:
        return [o for o in self._objects if o.round_number == round_number]

    def get_by_type(self, object_type: str) -> list[ResearchObject]:
        return [o for o in self._objects if o.object_type == object_type]

    def get_by_id(self, obj_id: str) -> ResearchObject | None:
        return self._index.get(obj_id)

    def get_scores(self) -> list[RoundScore]:
        return [o for o in self._objects if isinstance(o, RoundScore)]

    def save(self) -> None:
        self.ensure_dir()
        data = [obj.model_dump(mode="json") for obj in self._objects]
        self.memory_file.write_text(json.dumps(data, indent=2, default=str))
        hitl_path = self.session_dir / "hitl_review.json"
        hitl_path.write_text(json.dumps(self._hitl_entries, indent=2))

    def save_round(self, round_number: int) -> None:
        """Save a snapshot of a single round's objects."""
        self.ensure_dir()
        round_objs = self.get_by_round(round_number)
        data = [obj.model_dump(mode="json") for obj in round_objs]
        round_file = self.session_dir / f"round_{round_number:03d}.json"
        round_file.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        if not self.memory_file.exists():
            return
        raw = json.loads(self.memory_file.read_text())
        self._objects = [deserialize_research_object(item) for item in raw]
        self._index = {obj.id: obj for obj in self._objects}
        hitl_path = self.session_dir / "hitl_review.json"
        if hitl_path.exists():
            self._hitl_entries = json.loads(hitl_path.read_text())
        else:
            self._hitl_entries = []

    def save_config(self, config_data: dict) -> None:
        """Persist the session configuration alongside memory."""
        self.ensure_dir()
        import yaml

        config_file = self.session_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_data, default_flow_style=False, sort_keys=False))

    def get_session_metadata(self) -> dict:
        """Return metadata about this session for listing."""
        config_file = self.session_dir / "config.yaml"
        question = ""
        if config_file.exists():
            import yaml

            cfg = yaml.safe_load(config_file.read_text())
            question = cfg.get("question", "")

        scores = self.get_scores()
        rounds = max((o.round_number for o in self._objects), default=0) if self._objects else 0

        mtime = self.memory_file.stat().st_mtime if self.memory_file.exists() else 0
        modified = datetime.fromtimestamp(mtime, tz=timezone.utc) if mtime else None

        return {
            "session_id": self.session_id,
            "question": question,
            "rounds": rounds,
            "total_objects": len(self._objects),
            "last_verdict": scores[-1].verdict if scores else None,
            "modified": modified,
        }

    @classmethod
    def list_sessions(cls, sessions_dir: Path | None = None) -> list[dict]:
        """List all sessions in the sessions directory."""
        base = sessions_dir or DEFAULT_SESSIONS_DIR
        if not base.exists():
            return []

        sessions = []
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "memory.json").exists():
                store = cls(session_id=d.name, sessions_dir=base)
                store.load()
                sessions.append(store.get_session_metadata())
        return sessions
