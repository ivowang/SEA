"""Skill library as an evolution target.

Thin wrapper that delegates to SkillLibrary's Evolvable interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sea.agent.skills.library import SkillLibrary
from sea.core.base import Evolvable

logger = logging.getLogger(__name__)


class SkillTarget(Evolvable[list[dict[str, Any]]]):
    """Wraps a SkillLibrary as an explicit evolution target.

    Useful when the evolver needs to be explicitly wired to the skill library
    via the evolution pipeline config. Delegates to SkillLibrary's Evolvable.
    """

    def __init__(self, skill_library: SkillLibrary) -> None:
        self.skill_library = skill_library

    def get_evolvable_state(self) -> list[dict[str, Any]]:
        return self.skill_library.get_evolvable_state()

    def set_evolvable_state(self, state: list[dict[str, Any]]) -> None:
        self.skill_library.set_evolvable_state(state)

    def evolution_metadata(self) -> dict[str, Any]:
        return self.skill_library.evolution_metadata()

    def save_checkpoint(self, path: Path) -> None:
        self.skill_library.save_checkpoint(path)

    def load_checkpoint(self, path: Path) -> None:
        self.skill_library.load_checkpoint(path)

    def state_dict(self) -> dict[str, Any]:
        return self.skill_library.state_dict()
