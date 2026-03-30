"""Working memory: a sliding window over recent interaction steps."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from sea.agent.memory.base import Memory, MemoryEntry
from sea.core.registry import MEMORY_REGISTRY


@MEMORY_REGISTRY.register("working")
class WorkingMemory(Memory):
    """Fixed-size sliding window of recent entries.

    This is the simplest memory — it just keeps the last *max_size* entries
    in insertion order.  Retrieval returns entries by recency, ignoring the
    query.  Used as the agent's short-term context buffer.
    """

    def __init__(self, max_size: int = 20) -> None:
        self._buffer: deque[MemoryEntry] = deque(maxlen=max_size)
        self._max_size = max_size

    def add(self, entry: MemoryEntry) -> None:
        self._buffer.append(entry)

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Return recent entries, prioritizing those relevant to query."""
        import re
        entries = list(self._buffer)
        if not entries:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return entries[-k:]

        # Score by recency + keyword overlap
        scored = []
        for i, entry in enumerate(entries):
            content_words = set(re.findall(r"\w+", entry.content.lower()))
            overlap = len(query_words & content_words)
            recency = i / max(len(entries), 1)  # 0..1, higher = more recent
            score = overlap + recency * 0.5
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def get_all(self) -> list[MemoryEntry]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._buffer]
        (path / "working_memory.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "working_memory.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            self._buffer.clear()
            for d in data:
                self._buffer.append(MemoryEntry(**d))

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._buffer], "max_size": self._max_size}
