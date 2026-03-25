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
        # Return most recent k entries (newest last)
        entries = list(self._buffer)
        return entries[-k:]

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
