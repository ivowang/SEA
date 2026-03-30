"""Episodic memory: time-stamped buffer of experiences.

This memory is Evolvable — ICL evolvers can add reflections and exemplars.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sea.agent.memory.base import Memory, MemoryEntry
from sea.core.base import Evolvable
from sea.core.registry import MEMORY_REGISTRY

logger = logging.getLogger(__name__)


@MEMORY_REGISTRY.register("episodic")
class EpisodicMemory(Memory, Evolvable[list[dict[str, Any]]]):
    """Stores time-stamped episodic experiences.

    Retrieval is by recency and keyword overlap (simple TF matching).
    For embedding-based retrieval, use SemanticMemory instead.

    Implements Evolvable so ICL evolvers can add/modify memory contents.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_size = max_size

    def add(self, entry: MemoryEntry) -> None:
        if not entry.memory_type:
            entry.memory_type = "episodic"
        self._entries.append(entry)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size :]

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []
        import re
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for entry in self._entries:
            content_words = set(re.findall(r"\w+", entry.content.lower()))
            overlap = len(query_words & content_words)
            if overlap == 0:
                continue  # skip completely irrelevant entries
            # Boost by memory type and stored priority
            type_boost = 0.0
            if entry.memory_type in ("reflection", "semantic"):
                type_boost = 3.0
            # ExpeL-style priority stored in metadata
            priority = entry.metadata.get("priority", 0.0)
            if isinstance(priority, (int, float)):
                type_boost += float(priority)
            recency = entry.timestamp
            score = overlap + type_boost + recency * 1e-12
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

    # -- Evolvable --

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self._entries = [MemoryEntry(**d) for d in state]
        logger.info("Updated episodic memory: %d entries", len(self._entries))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "episodic_memory",
            "num_entries": len(self._entries),
            "max_size": self._max_size,
        }
