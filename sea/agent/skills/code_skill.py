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


class ComposedSkill(TextSkill):
    """A higher-level skill composed from atomic sub-skills.

    Represents a multi-step procedure built by combining simpler skills.
    E.g., "clean_and_place = navigate(obj) → pick(obj) → clean(obj) → put(obj, dest)"
    """

    def __init__(
        self,
        name: str,
        description: str,
        composition_plan: str,
        sub_skills: list[str],
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            instructions=composition_plan,
            tags=tags,
        )
        self._info.sub_skills = sub_skills
        self._info.composition_plan = composition_plan

    def to_prompt(self) -> str:
        """Concise prompt — just name and plan, not verbose instructions."""
        return f"[Composed Skill] {self.name}: {self.description}\n  Plan: {self._info.composition_plan}"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["sub_skills"] = self._info.sub_skills
        d["composition_plan"] = self._info.composition_plan
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposedSkill:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            composition_plan=data.get("composition_plan", data.get("instructions", "")),
            sub_skills=data.get("sub_skills", []),
            tags=data.get("tags", []),
        )
