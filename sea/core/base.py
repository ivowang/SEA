"""Foundation protocols for the SEA platform.

Every component that participates in checkpointing implements Checkpointable.
Every component that is a target of evolution implements Evolvable[T], where T
is the type of the state representation that evolvers operate on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Checkpointable(ABC):
    """Any component whose state can be saved and restored."""

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Persist current state to *path* (file or directory)."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Restore state from a previously saved checkpoint."""
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the current state."""
        ...


class Evolvable(Checkpointable, Generic[T]):
    """A component that is a target of evolution.

    The generic parameter *T* is the type of the state representation that
    evolvers read and write.  For example:

    * LoRA adapter  -> ``T = Path``   (path to adapter checkpoint)
    * Textual prompt -> ``T = str``
    * Memory         -> ``T = list[MemoryEntry]``
    * Skill library  -> ``T = list[dict]``
    """

    @abstractmethod
    def get_evolvable_state(self) -> T:
        """Return current state in a form that evolvers can manipulate."""
        ...

    @abstractmethod
    def set_evolvable_state(self, state: T) -> None:
        """Apply evolved state back to this component."""
        ...

    @abstractmethod
    def evolution_metadata(self) -> dict[str, Any]:
        """Return metadata about this evolution target.

        Examples: parameter count, target type, current version, etc.
        """
        ...
