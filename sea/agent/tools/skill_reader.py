"""ReadSkillTool: on-demand skill content retrieval for progressive disclosure.

Allows the agent to request the full content of a skill by name,
completing the 3-tier progressive disclosure pattern:
  INDEX (always in context) → SUMMARY (top-k retrieved) → FULL (this tool)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sea.agent.skills.disclosure import DisclosureLevel, view_to_prompt
from sea.agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from sea.agent.skills.library import SkillLibrary


class ReadSkillTool(Tool):
    """Read the full content of a named skill from the skill library."""

    def __init__(self, skill_library: SkillLibrary) -> None:
        self._skill_library = skill_library

    @property
    def name(self) -> str:
        return "read_skill"

    @property
    def description(self) -> str:
        return "Read the full content of a skill by name. Use when you need detailed steps for a specific skill."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to read",
                },
            },
            "required": ["skill_name"],
        }

    def execute(self, skill_name: str = "", **kwargs: Any) -> ToolResult:
        view = self._skill_library.retrieve_full(skill_name)
        if view is None:
            available = [s.name for s in self._skill_library.list_skills()]
            return ToolResult(
                output=f"Skill '{skill_name}' not found. Available: {available}",
                success=False,
            )
        return ToolResult(output=view_to_prompt(view))
