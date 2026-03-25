"""Episodic memory: time-stamped buffer of experiences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sea.agent.memory.base import Memory, MemoryEntry
from sea.core.registry import MEMORY_REGISTRY


@MEMORY_REGISTRY.register("episodic")
class EpisodicMemory(Memory):
    """Stores time-stamped episodic experiences.

    Retrieval is by recency and keyword overlap (simple TF matching).
    For embedding-based retrieval, use SemanticMemory instead.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_size = max_size

    def add(self, entry: MemoryEntry) -> None:
        entry.memory_type = "episodic"
        self._entries.append(entry)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size :]

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []
        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            # Combine keyword overlap with recency
            recency = entry.timestamp
            score = overlap + recency * 1e-12
            entry.score = score
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        (path / "episodic_memory.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "episodic_memory.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            self._entries = [MemoryEntry(**d) for d in data]

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._entries], "max_size": self._max_size}
