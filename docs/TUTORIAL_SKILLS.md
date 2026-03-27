# Tutorial: Skill Evolution on TextCraft

Write a custom `Evolver` that extracts reusable Minecraft sub-recipes from successful TextCraft trajectories. Skills like "craft_oak_planks" and "gray_dye" are stored in the SkillLibrary and retrieved for new tasks sharing sub-recipes.

**No GPU required** — uses any OpenAI-compatible API.

## Quick Run

```bash
export SEA_API_KEY="your-key"
export SEA_BASE_URL="https://api.example.com/v1"
export SEA_MODEL="openai/gpt-5.4-nano"
python examples/skill_textcraft/run.py
```

## Results on TextCraft (gpt-5.4-nano)

| Stage | Success Rate | Skills |
|-------|-------------|--------|
| Baseline | 70% | 0 |
| Iter 1 | 50% | 3 |
| Iter 2 | 40% | 4 |
| Iter 3 | 60% | 6 |
| Iter 4 | 50% | 9 |

**9 Minecraft sub-recipes extracted:**
- `craft_oak_planks` — Produces oak planks from oak logs
- `purple_dye` — Produces purple dye from basic dyes
- `gray_dye` — Produces 2 gray dye
- `craft_andesite` — Produces andesite from diorite and cobblestone
- `polished_granite` — Craft polished granite from granite
- ... and 4 more

These are real, reusable Minecraft crafting patterns.

## The Custom SkillExtractEvolver (~40 lines)

This is the core of the tutorial — implementing a new `Evolver`:

```python
from sea.core.registry import EVOLVER_REGISTRY
from sea.evolution.base import Evolver
from sea.agent.skills.code_skill import TextSkill

@EVOLVER_REGISTRY.register("skill_extract_tc")
class SkillExtractEvolver(Evolver):

    def __init__(self, max_skills_per_iter: int = 3):
        self._max_skills = max_skills_per_iter

    def requires_trajectories(self):
        return True

    def evolve(self, agent, target, trajectories, metrics, **kwargs):
        positive = sorted(
            [t for t in trajectories if t.total_reward > 0],
            key=lambda t: t.total_reward, reverse=True,
        )
        if not positive:
            return

        existing = {s.name for s in agent.skill_library.list_skills()}

        for traj in positive[:self._max_skills]:
            steps_text = " → ".join(s.action.text.strip() for s in traj.steps)

            # Ask LLM to extract a reusable sub-recipe
            messages = [
                {"role": "system", "content":
                    "Extract ONE reusable Minecraft sub-recipe from this trajectory.\n"
                    "Respond as:\nSKILL_NAME: <name>\nDESCRIPTION: <desc>\nSTEPS: <steps>"},
                {"role": "user", "content": f"Goal: {traj.metadata.get('task_description','')}\nActions: {steps_text}"},
            ]
            output = agent.brain.generate(messages, temperature=0.3, max_tokens=150)

            # Parse response
            name = desc = steps = ""
            for line in output.text.strip().split("\n"):
                if line.upper().startswith("SKILL_NAME:"): name = line.split(":",1)[1].strip().lower().replace(" ","_")
                elif line.upper().startswith("DESCRIPTION:"): desc = line.split(":",1)[1].strip()
                elif line.upper().startswith("STEPS:"): steps = line.split(":",1)[1].strip()

            if not name or name in existing:
                continue

            skill = TextSkill(name=name, description=desc, instructions=f"Steps: {steps}")
            agent.skill_library.add_skill(skill)
            existing.add(name)
```

## Key Points

1. **Custom Evolvers are ~40 lines**: Inherit `Evolver`, implement `evolve()`, register with `@EVOLVER_REGISTRY`
2. **SkillLibrary truthiness**: Empty `SkillLibrary` is falsy — always use `is not None` checks
3. **Pre-warm embedding model**: Call `agent.skill_library._ensure_loaded()` at startup to avoid slow first `add_skill()`
4. **TextCraft skills are real**: The extracted sub-recipes correspond to actual Minecraft crafting patterns

## Full Script

See `examples/skill_textcraft/run.py`
