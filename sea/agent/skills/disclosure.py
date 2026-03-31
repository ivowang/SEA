"""Progressive Disclosure for skills.

Three-tier disclosure model:
- INDEX: name + description (lightweight, always in context)
- SUMMARY: + when_to_use + steps outline (top-k retrieved)
- FULL: complete markdown body (on-demand)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sea.agent.skills.skill_md import SkillMd


class DisclosureLevel(Enum):
    """How much of a skill to disclose to the agent."""

    INDEX = "index"
    SUMMARY = "summary"
    FULL = "full"


@dataclass
class SkillView:
    """A view of a skill at a specific disclosure level."""

    name: str
    description: str
    level: DisclosureLevel
    when_to_use: str = ""
    steps_outline: str = ""
    full_content: str = ""
    tags: list[str] = field(default_factory=list)
    sub_skills: list[str] = field(default_factory=list)


def _extract_steps_outline(body: str, max_items: int = 5) -> str:
    """Extract the first few items from a ## Steps section, or first lines."""
    # Try to find ## Steps section
    steps_match = re.search(r"##\s*Steps\s*\n(.*?)(?:\n##|\Z)", body, re.DOTALL)
    if steps_match:
        section = steps_match.group(1).strip()
        lines = [l.strip() for l in section.split("\n") if l.strip()]
        return "\n".join(lines[:max_items])

    # Fallback: first N non-empty lines of body
    lines = [l.strip() for l in body.split("\n") if l.strip() and not l.startswith("#")]
    return "\n".join(lines[:max_items])


def skill_to_view(skill: SkillMd, level: DisclosureLevel) -> SkillView:
    """Create a SkillView at the specified disclosure level."""
    fm = skill.frontmatter
    view = SkillView(
        name=fm.name,
        description=fm.description,
        level=level,
        tags=fm.tags,
        sub_skills=fm.sub_skills,
    )

    if level in (DisclosureLevel.SUMMARY, DisclosureLevel.FULL):
        view.when_to_use = fm.when_to_use
        view.steps_outline = _extract_steps_outline(skill.body)

    if level == DisclosureLevel.FULL:
        view.full_content = skill.body

    return view


def view_to_prompt(view: SkillView) -> str:
    """Format a SkillView for injection into the LLM prompt."""
    if view.level == DisclosureLevel.INDEX:
        return f"- {view.name}: {view.description}"

    parts = [f"Skill: {view.name}", f"Description: {view.description}"]

    if view.when_to_use:
        parts.append(f"When to use: {view.when_to_use}")

    if view.steps_outline:
        parts.append(f"Steps:\n{view.steps_outline}")

    if view.level == DisclosureLevel.FULL and view.full_content:
        parts.append(f"Full details:\n{view.full_content}")

    if view.sub_skills:
        parts.append(f"Uses sub-skills: {', '.join(view.sub_skills)}")

    return "\n".join(parts)
