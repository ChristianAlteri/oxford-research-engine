"""Memory store — JSON-backed persistence for research objects and sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ore.research_objects import ResearchObject, RoundScore, deserialize_research_object


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

    def __init__(self, session_id: str | None = None, sessions_dir: Path | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.sessions_dir = sessions_dir or DEFAULT_SESSIONS_DIR
        self.session_dir = self.sessions_dir / self.session_id
        self._objects: list[ResearchObject] = []
        self._index: dict[str, ResearchObject] = {}

    @property
    def memory_file(self) -> Path:
        return self.session_dir / "memory.json"

    def ensure_dir(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def add(self, obj: ResearchObject) -> None:
        self._objects.append(obj)
        self._index[obj.id] = obj

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
