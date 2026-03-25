"""Code-based skills (Voyager pattern): executable Python as skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sea.agent.skills.base import Skill, SkillInfo


class CodeSkill(Skill):
    """A skill stored as executable Python code.

    Inspired by Voyager's skill library: each skill is a Python function
    that can be executed in a sandboxed namespace.
    """

    def __init__(
        self,
        name: str,
        description: str,
        source_code: str,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
    ) -> None:
        self._info = SkillInfo(
            name=name,
            description=description,
            tags=tags or [],
            examples=examples or [],
        )
        self.source_code = source_code

    @property
    def info(self) -> SkillInfo:
        return self._info

    def to_prompt(self) -> str:
        return (
            f"Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Code:\n```python\n{self.source_code}\n```"
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["source_code"] = self.source_code
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeSkill:
        return cls(
            name=data["name"],
            description=data["description"],
            source_code=data["source_code"],
            tags=data.get("tags", []),
            examples=data.get("examples", []),
        )

    def execute_in_sandbox(self, **kwargs: Any) -> Any:
        """Execute the skill code in a restricted namespace."""
        namespace: dict[str, Any] = {"__builtins__": {}, **kwargs}
        exec(self.source_code, namespace)  # noqa: S102
        main_fn = namespace.get("main") or namespace.get(self.name)
        if callable(main_fn):
            return main_fn(**kwargs)
        return namespace


class TextSkill(Skill):
    """A skill stored as natural language instructions."""

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
    ) -> None:
        self._info = SkillInfo(
            name=name,
            description=description,
            tags=tags or [],
            examples=examples or [],
        )
        self.instructions = instructions

    @property
    def info(self) -> SkillInfo:
        return self._info

    def to_prompt(self) -> str:
        return (
            f"Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Instructions:\n{self.instructions}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["instructions"] = self.instructions
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextSkill:
        return cls(
            name=data["name"],
            description=data["description"],
            instructions=data["instructions"],
            tags=data.get("tags", []),
            examples=data.get("examples", []),
        )
