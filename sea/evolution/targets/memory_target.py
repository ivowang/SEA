"""Memory as an evolution target.

Wraps a Memory instance and exposes its contents for evolution —
evolvers can curate, compress, or reorganise the memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sea.agent.memory.base import Memory, MemoryEntry
from sea.core.base import Evolvable

logger = logging.getLogger(__name__)


class MemoryTarget(Evolvable[list[dict[str, Any]]]):
    """Wraps a Memory instance as an explicit evolution target.

    Use this when you want an evolver to directly manipulate memory contents
    (e.g., curate reflections, compress episodic memories into semantic ones).

    The evolvable state is the list of serialised MemoryEntry dicts.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self._version = 0

    # -- Evolvable --

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.memory.get_all()]

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self.memory.clear()
        for d in state:
            self.memory.add(MemoryEntry(
                content=d["content"],
                metadata=d.get("metadata", {}),
                timestamp=d.get("timestamp", 0.0),
                memory_type=d.get("memory_type", "semantic"),
            ))
        self._version += 1
        logger.info("Memory target updated: %d entries (version %d)",
                     len(state), self._version)

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "memory",
            "num_entries": self.memory.size(),
            "version": self._version,
            "memory_class": type(self.memory).__name__,
        }

    # -- Checkpointable (delegates to wrapped memory) --

    def save_checkpoint(self, path: Path) -> None:
        self.memory.save_checkpoint(path)

    def load_checkpoint(self, path: Path) -> None:
        self.memory.load_checkpoint(path)

    def state_dict(self) -> dict[str, Any]:
        return self.evolution_metadata()
