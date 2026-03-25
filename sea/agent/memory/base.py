"""Abstract base for memory and shared MemoryEntry dataclass."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sea.core.base import Checkpointable


@dataclass
class MemoryEntry:
    """A single memory record."""

    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    memory_type: str = "episodic"  # "episodic", "semantic", "reflection", "skill"
    score: float = 0.0  # relevance score (filled during retrieval)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
        }


class Memory(Checkpointable):
    """Abstract memory interface.

    All memory types (episodic, semantic, working) implement this ABC.
    Memory instances that are evolution targets should additionally
    implement Evolvable[list[MemoryEntry]].
    """

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Store a new memory entry."""
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Retrieve the top-k most relevant entries for *query*."""
        ...

    @abstractmethod
    def get_all(self) -> list[MemoryEntry]:
        """Return all stored entries."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        ...

    def size(self) -> int:
        """Number of entries currently stored."""
        return len(self.get_all())
