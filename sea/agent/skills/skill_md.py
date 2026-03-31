"""SKILL.md format: YAML frontmatter + Markdown body.

Each skill is a .md file with structured metadata in YAML frontmatter
and step-by-step instructions in the markdown body. This is the modern
Progressive Disclosure paradigm for agent skills.

Example SKILL.md:
    ---
    name: craft_oak_planks
    description: Craft oak planks from oak logs
    version: 1
    tags: [crafting, textcraft]
    when_to_use: When you need oak planks and have oak logs
    sub_skills: []
    ---

    ## Steps
    1. Check inventory: `inventory`
    2. Get logs: `get 1 oak log`
    3. Craft: `craft 4 oak planks using 1 oak log`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillFrontmatter:
    """Structured metadata for a skill (YAML frontmatter)."""

    name: str
    description: str
    version: int = 1
    tags: list[str] = field(default_factory=list)
    when_to_use: str = ""
    sub_skills: list[str] = field(default_factory=list)


@dataclass
class SkillMd:
    """A parsed SKILL.md file: frontmatter + markdown body."""

    frontmatter: SkillFrontmatter
    body: str  # markdown content after frontmatter
    file_path: Path | None = None

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description


# ---------------------------------------------------------------------------
# Parse / Render
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_skill_md(text: str) -> SkillMd:
    """Parse a SKILL.md string into a SkillMd object.

    Raises ValueError if the frontmatter is missing or invalid.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("SKILL.md must start with YAML frontmatter (--- ... ---)")

    raw_yaml = m.group(1)
    body = text[m.end():]

    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"Frontmatter must be a YAML mapping, got {type(data).__name__}")
    if "name" not in data:
        raise ValueError("Frontmatter must contain 'name' field")

    fm = SkillFrontmatter(
        name=data["name"],
        description=data.get("description", ""),
        version=data.get("version", 1),
        tags=data.get("tags", []),
        when_to_use=data.get("when_to_use", ""),
        sub_skills=data.get("sub_skills", []),
    )
    return SkillMd(frontmatter=fm, body=body.strip())


def render_skill_md(skill: SkillMd) -> str:
    """Render a SkillMd object back to a SKILL.md string."""
    fm = skill.frontmatter
    data: dict[str, Any] = {
        "name": fm.name,
        "description": fm.description,
    }
    if fm.version != 1:
        data["version"] = fm.version
    if fm.tags:
        data["tags"] = fm.tags
    if fm.when_to_use:
        data["when_to_use"] = fm.when_to_use
    if fm.sub_skills:
        data["sub_skills"] = fm.sub_skills

    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yaml_str}\n---\n\n{skill.body}\n"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_skill_md(path: Path) -> SkillMd:
    """Load and parse a SKILL.md file."""
    text = path.read_text(encoding="utf-8")
    skill = parse_skill_md(text)
    skill.file_path = path
    return skill


def save_skill_md(skill: SkillMd, path: Path) -> None:
    """Render and write a SkillMd to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_skill_md(skill), encoding="utf-8")
    skill.file_path = path


# ---------------------------------------------------------------------------
# Legacy conversion (JSON dict ↔ SkillMd)
# ---------------------------------------------------------------------------

def skill_from_dict(d: dict[str, Any]) -> SkillMd:
    """Convert a legacy skill dict (from old JSON format) to SkillMd.

    Handles TextSkill, CodeSkill, and ComposedSkill dicts.
    """
    name = d.get("name", "unnamed")
    description = d.get("description", "")
    tags = d.get("tags", [])
    sub_skills = d.get("sub_skills", [])

    # Build body from the skill content
    body_parts: list[str] = []

    # ComposedSkill
    if "composition_plan" in d:
        if d.get("composition_plan"):
            body_parts.append(f"## Plan\n{d['composition_plan']}")
        if sub_skills:
            body_parts.append("## Sub-skills\n" + "\n".join(f"- {s}" for s in sub_skills))

    # CodeSkill
    elif "source_code" in d:
        body_parts.append(f"## Code\n```python\n{d['source_code']}\n```")

    # TextSkill
    elif "instructions" in d:
        body_parts.append(d["instructions"])

    # Fallback: use description
    if not body_parts:
        body_parts.append(description)

    # Add examples if present
    examples = d.get("examples", [])
    if examples:
        body_parts.append("## Examples\n" + "\n".join(f"- {ex}" for ex in examples))

    fm = SkillFrontmatter(
        name=name,
        description=description,
        tags=tags,
        sub_skills=sub_skills,
    )
    return SkillMd(frontmatter=fm, body="\n\n".join(body_parts))


def skill_to_dict(skill: SkillMd) -> dict[str, Any]:
    """Convert SkillMd to legacy dict format (for Evolvable contract)."""
    fm = skill.frontmatter
    d: dict[str, Any] = {
        "name": fm.name,
        "description": fm.description,
        "tags": fm.tags,
        "type": "SkillMd",
    }
    if fm.sub_skills:
        d["sub_skills"] = fm.sub_skills
        d["composition_plan"] = skill.body
    d["instructions"] = skill.body
    if fm.when_to_use:
        d["when_to_use"] = fm.when_to_use
    return d


def _sanitize_filename(name: str) -> str:
    """Convert a skill name to a safe filename (without extension)."""
    return re.sub(r"[^a-z0-9_-]", "_", name.lower()).strip("_")
