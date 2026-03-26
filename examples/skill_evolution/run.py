#!/usr/bin/env python3
"""Skills Evolution Demo: Learning reusable skills in Recipe Kitchen.

The agent faces cooking tasks that share common sub-steps. A custom
SkillEvolver extracts reusable skills from successful trajectories and
stores them in the SkillLibrary. In subsequent iterations, retrieved
skills help the agent complete new tasks faster.

Usage:
    export SEA_API_KEY="your-api-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/skill_evolution/run.py

To use a local vLLM server instead:
    python scripts/serve_model.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
    export SEA_BASE_URL="http://localhost:8000/v1"
    export SEA_MODEL="Qwen/Qwen2.5-7B-Instruct"
    python examples/skill_evolution/run.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("skill_evolution")

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 12

# ──────────────────────────────────────────────────────────────────────
# 1. Recipe Kitchen Environment
# ──────────────────────────────────────────────────────────────────────
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

RECIPES = {
    "sandwich": {
        "description": "Make a sandwich. You need bread, lettuce, cheese, and assemble them.",
        "steps": ["get bread", "add lettuce", "add cheese", "close sandwich"],
    },
    "salad": {
        "description": "Make a salad. You need a bowl, lettuce, tomato, and mix them.",
        "steps": ["get bowl", "add lettuce", "add tomato", "mix"],
    },
    "soup": {
        "description": "Make soup. You need a pot, water, tomato, and cook it.",
        "steps": ["get pot", "add water", "add tomato", "boil"],
    },
    "pasta": {
        "description": "Make pasta. You need a pot, water, boil it, add pasta, and drain.",
        "steps": ["get pot", "add water", "boil", "add pasta", "drain"],
    },
    "smoothie": {
        "description": "Make a smoothie. You need a blender, banana, milk, and blend.",
        "steps": ["get blender", "add banana", "add milk", "blend"],
    },
    "omelette": {
        "description": "Make an omelette. You need a pan, eggs, cheese, and cook.",
        "steps": ["get pan", "crack eggs", "add cheese", "cook"],
    },
    "stew": {
        "description": "Make stew. You need a pot, water, potato, carrot, and simmer.",
        "steps": ["get pot", "add water", "add potato", "add carrot", "simmer"],
    },
    "tea": {
        "description": "Make tea. You need a kettle, water, boil it, get cup, add tea bag, and pour.",
        "steps": ["get kettle", "add water", "boil", "get cup", "add tea bag", "pour water"],
    },
}


class RecipeKitchenEnv(SEAEnv):
    """Cooking tasks with shared sub-steps across recipes."""

    def __init__(self):
        self._recipe = None
        self._recipe_id = ""
        self._completed = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "recipe_kitchen"

    @property
    def max_steps(self) -> int:
        return 12

    def get_task_ids(self) -> list[str]:
        return list(RECIPES.keys())

    def reset(self, *, seed=None, task_id=None):
        if seed is not None:
            random.seed(seed)
        self._recipe_id = task_id or random.choice(list(RECIPES.keys()))
        self._recipe = RECIPES[self._recipe_id]
        self._completed = []
        self._step_count = 0

        obs = (
            f"Kitchen Task: {self._recipe['description']}\n"
            f"Steps needed: {len(self._recipe['steps'])}. What do you do first?"
        )
        return Observation(text=obs), {
            "task_id": self._recipe_id,
            "task_description": self._recipe["description"],
        }

    @staticmethod
    def _fuzzy_match(expected: str, action: str) -> bool:
        """Check if all keywords in expected appear in action."""
        keywords = expected.lower().split()
        action_lower = action.lower()
        return all(kw in action_lower for kw in keywords)

    def step(self, action):
        self._step_count += 1
        action_lower = action.text.lower()

        steps = self._recipe["steps"]
        next_idx = len(self._completed)

        # Match next expected step
        if next_idx < len(steps):
            expected = steps[next_idx]
            if self._fuzzy_match(expected, action_lower):
                self._completed.append(expected)
                all_done = len(self._completed) >= len(steps)
                reward = 0.2 + (0.5 if all_done else 0.0)
                obs = f"Done: {expected}. " + (
                    f"Recipe complete! ({self._recipe_id})"
                    if all_done
                    else f"Progress: {len(self._completed)}/{len(steps)}."
                )
                return Observation(text=obs), reward, all_done, False, {"success": all_done}

        # Allow matching any remaining step (flexible order for some sub-steps)
        for i in range(next_idx, len(steps)):
            if self._fuzzy_match(steps[i], action_lower) and steps[i] not in self._completed:
                self._completed.append(steps[i])
                all_done = len(self._completed) >= len(steps)
                reward = 0.15 + (0.5 if all_done else 0.0)
                obs = f"Done: {steps[i]} (out of order). Progress: {len(self._completed)}/{len(steps)}."
                if all_done:
                    obs = f"Done: {steps[i]}. Recipe complete! ({self._recipe_id})"
                return Observation(text=obs), reward, all_done, False, {"success": all_done}

        # No match
        truncated = self._step_count >= self.max_steps
        obs = (
            f"That's not a valid cooking step. "
            f"Progress: {len(self._completed)}/{len(steps)}. "
            f"Hint: next step involves '{steps[next_idx]}' (use exact keywords)."
        )
        return Observation(text=obs), 0.0, False, truncated, {"success": False}


# ──────────────────────────────────────────────────────────────────────
# 2. Custom SkillEvolver
# ──────────────────────────────────────────────────────────────────────
from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.agent.skills.code_skill import TextSkill
from sea.metrics.tracker import MetricsTracker


@EVOLVER_REGISTRY.register("skill_extract")
class SkillExtractEvolver(Evolver):
    """Extracts reusable skills from successful trajectories.

    For each successful trajectory, asks the LLM to identify sub-step
    patterns that could be reused in other recipes, and stores them
    as TextSkill in the SkillLibrary.
    """

    def __init__(self, max_skills_per_iter: int = 3):
        self._max_skills = max_skills_per_iter

    def requires_trajectories(self) -> bool:
        return True

    def evolve(self, agent, target, trajectories, metrics):
        successful = [t for t in trajectories if t.success]
        if not successful:
            logger.info("SkillEvolver: no successful trajectories, skipping")
            return

        skills_added = 0
        existing_names = {s.name for s in agent.skill_library.list_skills()}

        for traj in successful[:self._max_skills]:
            # Build step summary
            steps_text = " → ".join(s.action.text.strip() for s in traj.steps if s.action.text.strip())
            task_id = traj.task_id

            # Ask LLM to extract a reusable skill
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a successful cooking trajectory. "
                        "Extract ONE reusable sub-procedure that could help with OTHER recipes. "
                        "Focus on common patterns like 'boil water', 'prepare ingredients', etc.\n\n"
                        "Respond in EXACTLY this format:\n"
                        "SKILL_NAME: <short_name_with_underscores>\n"
                        "DESCRIPTION: <one line description>\n"
                        "STEPS: <step1> -> <step2> -> <step3>"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Recipe: {task_id}\nActions taken: {steps_text}",
                },
            ]

            output = agent.brain.generate(messages, temperature=0.3, max_tokens=150)
            text = output.text.strip()

            # Parse response
            skill_name = ""
            description = ""
            steps = ""
            for line in text.split("\n"):
                line = line.strip()
                if line.upper().startswith("SKILL_NAME:"):
                    skill_name = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                elif line.upper().startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()
                elif line.upper().startswith("STEPS:"):
                    steps = line.split(":", 1)[1].strip()

            if not skill_name or skill_name in existing_names:
                continue

            skill = TextSkill(
                name=skill_name,
                description=description or f"Skill extracted from {task_id}",
                instructions=f"Steps: {steps}" if steps else text,
                tags=["auto_extracted", task_id],
            )
            agent.skill_library.add_skill(skill)
            existing_names.add(skill_name)
            skills_added += 1
            logger.info("  Extracted skill: '%s' from %s", skill_name, task_id)

        metrics.log({
            "skills/added": skills_added,
            "skills/total": len(agent.skill_library),
        })
        logger.info("SkillEvolver: +%d skills (%d total)", skills_added, len(agent.skill_library))


# ──────────────────────────────────────────────────────────────────────
# 3. Build agent
# ──────────────────────────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.agent.skills.library import SkillLibrary
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator
from sea.evolution.data.trajectory import TrajectoryCollector

logger.info("=" * 60)
logger.info("Skills Evolution Demo: Recipe Kitchen")
logger.info("=" * 60)

backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a kitchen assistant that follows recipes step by step. "
            "Each recipe requires specific actions in order. "
            "Use the EXACT keywords from the recipe description. "
            "For example, if the recipe says 'add lettuce', your action should contain 'add lettuce'."
        ),
        default_max_tokens=150,
        default_temperature=0.7,
    ),
    memory=WorkingMemory(max_size=10),
    planner=ReActPlanner(),
    skill_library=SkillLibrary(),
)

env = RecipeKitchenEnv()
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()
evolver = SkillExtractEvolver(max_skills_per_iter=3)

# ──────────────────────────────────────────────────────────────────────
# 4. Baseline
# ──────────────────────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ──────────────────────────────────────────────────────────────────────
# 5. Evolution loop
# ──────────────────────────────────────────────────────────────────────
results_history = [("baseline", baseline.success_rate, baseline.avg_reward, 0)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (skills: %d)",
                iteration, NUM_ITERATIONS, len(agent.skill_library))
    logger.info("=" * 60)

    # Collect
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    success_count = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", success_count, len(trajectories))

    # Evolve skills
    skill_target = agent.evolvable_components().get("skill_library")
    if skill_target:
        evolver.evolve(agent, skill_target, trajectories, metrics)

    # List current skills
    for skill in agent.skill_library.list_skills():
        logger.info("  Skill: '%s' — %s", skill.name, skill.description[:60])

    # Evaluate
    result = evaluator.evaluate(agent, [env])
    n_skills = len(agent.skill_library)
    results_history.append((f"iter_{iteration}", result.success_rate, result.avg_reward, n_skills))
    logger.info(
        "Iter %d: success=%.0f%%, reward=%.3f, steps=%.1f, skills=%d",
        iteration, result.success_rate * 100, result.avg_reward, result.avg_steps, n_skills,
    )

# ──────────────────────────────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Evolution Summary")
logger.info("=" * 60)
for name, sr, ar, ns in results_history:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f  skills=%d", name, sr * 100, ar, ns)

improvement = results_history[-1][1] - results_history[0][1]
logger.info("Improvement: %+.0f%% success rate", improvement * 100)

# List final skill library
logger.info("Final Skill Library (%d skills):", len(agent.skill_library))
for skill in agent.skill_library.list_skills():
    logger.info("  [%s] %s", skill.name, skill.description)

# Save
import json
output_dir = Path("outputs/skill_evolution")
output_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "model": MODEL,
    "num_iterations": NUM_ITERATIONS,
    "results": [{"stage": n, "success_rate": s, "avg_reward": r, "num_skills": ns}
                for n, s, r, ns in results_history],
    "final_skills": [s.to_dict() for s in agent.skill_library.list_skills()],
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
logger.info("Summary saved to %s", output_dir / "summary.json")
