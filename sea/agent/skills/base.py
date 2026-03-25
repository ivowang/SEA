"""Abstract base for skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sea.core.types import Action


@dataclass
class SkillInfo:
    """Metadata describing a skill."""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """A reusable capability the agent has learned.

    Skills can be text-based instructions or executable code.
    """

    @property
    @abstractmethod
    def info(self) -> SkillInfo:
        """Return skill metadata."""
        ...

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def description(self) -> str:
        return self.info.description

    @abstractmethod
    def to_prompt(self) -> str:
        """Convert this skill into a textual prompt for the LLM."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise skill to a dictionary."""
        return {
            "name": self.info.name,
            "description": self.info.description,
            "tags": self.info.tags,
            "examples": self.info.examples,
            "type": self.__class__.__name__,
        }
