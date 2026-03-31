# Tutorial 2: Skill Evolution (SKILL.md Progressive Disclosure) on TextCraft

Extract reusable crafting skills from successful TextCraft trajectories. Skills are stored as **SKILL.md files** with YAML frontmatter and progressive disclosure.

**No GPU required** — uses OpenAI-compatible API (`gpt-5.4-nano`).

## Quick Start

```bash
python examples/skill_textcraft/run.py
```

## The SKILL.md Format

Each extracted skill is a markdown file:
```markdown
---
name: craft_oak_planks
description: Craft oak planks from oak logs
tags: [crafting, textcraft]
when_to_use: When you need oak planks and have oak logs
---

## Steps
1. Check inventory: `inventory`
2. Get logs: `get 1 oak log`
3. Craft: `craft 4 oak planks using 1 oak log`
```

## Progressive Disclosure (3 Tiers)

| Level | Agent Sees | When |
|-------|-----------|------|
| **INDEX** | name + description | Always in system prompt |
| **SUMMARY** | + when_to_use + steps outline | Top-k retrieved |
| **FULL** | Complete .md body | On-demand via `read_skill` tool |

## How It Works

1. **Collect** 30 trajectories on TextCraft
2. **SkillExtractEvolver** (custom ~50 line evolver):
   - Filter successful trajectories
   - LLM extracts reusable sub-recipes in SKILL.md format
   - Dedup by description overlap
   - `add_skill()` writes `.md` files to disk
3. **Evaluate** with skills injected via progressive disclosure

## Key Components

```python
# File-system backed skill library
lib = SkillLibrary(skills_dir="outputs/skills/", use_embeddings=False)

# Progressive disclosure API
lib.get_index()         # → list[SkillView] at INDEX level
lib.retrieve(query, k)  # → list[SkillView] at SUMMARY level
lib.retrieve_full(name) # → SkillView at FULL level
```

## Expected Results

| Stage | Success Rate | Skills |
|-------|-------------|--------|
| Baseline | ~70-80% | 0 |
| Iter 5 | ~85-92% | 12-18 |

Improvement: **+10-15%**. Skills like `craft_oak_planks`, `make_sticks` emerge and transfer.
