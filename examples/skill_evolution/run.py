#!/usr/bin/env python3
"""Skills Evolution Demo: Learning reusable skills in Recipe Kitchen.

The agent faces cooking tasks that share common sub-steps. A custom
SkillExtractEvolver extracts reusable skills from successful trajectories
and stores them in the SkillLibrary. In later iterations, retrieved
skills help the agent complete new tasks faster.

Usage:
    export SEA_API_KEY="your-api-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/skill_evolution/run.py

To use a local vLLM server instead:
    export SEA_BASE_URL="http://localhost:8000/v1"
    export SEA_MODEL="Qwen/Qwen2.5-7B-Instruct"
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("skill_evolution")

# ── Config ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 12

# ── 1. Recipe Kitchen Environment ─────────────────────────────────────
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

RECIPES = {
    "sandwich": {
        "description": "Make a sandwich: get bread, add lettuce, add cheese, close sandwich.",
        "steps": ["get bread", "add lettuce", "add cheese", "close sandwich"],
    },
    "salad": {
        "description": "Make a salad: get bowl, add lettuce, add tomato, mix.",
        "steps": ["get bowl", "add lettuce", "add tomato", "mix"],
    },
    "soup": {
        "description": "Make soup: get pot, add water, add tomato, boil.",
        "steps": ["get pot", "add water", "add tomato", "boil"],
    },
    "pasta": {
        "description": "Make pasta: get pot, add water, boil, add pasta, drain.",
        "steps": ["get pot", "add water", "boil", "add pasta", "drain"],
    },
    "smoothie": {
        "description": "Make a smoothie: get blender, add banana, add milk, blend.",
        "steps": ["get blender", "add banana", "add milk", "blend"],
    },
    "omelette": {
        "description": "Make an omelette: get pan, crack eggs, add cheese, cook.",
        "steps": ["get pan", "crack eggs", "add cheese", "cook"],
    },
    "stew": {
        "description": "Make stew: get pot, add water, add potato, add carrot, simmer.",
        "steps": ["get pot", "add water", "add potato", "add carrot", "simmer"],
    },
    "tea": {
        "description": "Make tea: get kettle, add water, boil, get cup, add tea bag, pour water.",
        "steps": ["get kettle", "add water", "boil", "get cup", "add tea bag", "pour water"],
    },
}


class RecipeKitchenEnv(SEAEnv):
    """Cooking tasks with shared sub-steps across recipes."""

    def __init__(self):
        self._recipe = None
        self._recipe_id = ""
        self._completed: list[str] = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "recipe_kitchen"

    @property
    def max_steps(self) -> int:
        return 12

    def get_task_ids(self) -> list[str]:
        return list(RECIPES.keys())

    @staticmethod
    def _fuzzy_match(expected: str, action: str) -> bool:
        keywords = expected.lower().split()
        return all(kw in action.lower() for kw in keywords)

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

    def step(self, action):
        self._step_count += 1
        steps = self._recipe["steps"]
        next_idx = len(self._completed)

        # Match next expected step (in order)
        if next_idx < len(steps) and self._fuzzy_match(steps[next_idx], action.text):
            self._completed.append(steps[next_idx])
            all_done = len(self._completed) >= len(steps)
            reward = 0.2 + (0.5 if all_done else 0.0)
            obs = f"Done: {steps[next_idx]}."
            if all_done:
                obs += f" Recipe complete! ({self._recipe_id})"
            else:
                obs += f" ({len(self._completed)}/{len(steps)})"
            return Observation(text=obs), reward, all_done, False, {"success": all_done}

        # Also allow matching any remaining step (out of order)
        for i in range(next_idx, len(steps)):
            if self._fuzzy_match(steps[i], action.text) and steps[i] not in self._completed:
                self._completed.append(steps[i])
                all_done = len(self._completed) >= len(steps)
                reward = 0.15 + (0.5 if all_done else 0.0)
                obs = f"Done: {steps[i]} (out of order). ({len(self._completed)}/{len(steps)})"
                if all_done:
                    obs = f"Done: {steps[i]}. Recipe complete! ({self._recipe_id})"
                return Observation(text=obs), reward, all_done, False, {"success": all_done}

        truncated = self._step_count >= self.max_steps
        obs = (
            f"Not a valid step. ({len(self._completed)}/{len(steps)} done). "
            f"Next: '{steps[next_idx]}' — use exact keywords."
        )
        return Observation(text=obs), 0.0, False, truncated, {"success": False}


# ── 2. Custom SkillExtractEvolver ─────────────────────────────────────
from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.agent.skills.code_skill import TextSkill


@EVOLVER_REGISTRY.register("skill_extract")
class SkillExtractEvolver(Evolver):
    """Extracts reusable skills from successful trajectories via LLM.

    This is a custom Evolver (~40 lines of logic) demonstrating how
    researchers can implement their own evolution methods on SEA.
    """

    def __init__(self, max_skills_per_iter: int = 3):
        self._max_skills = max_skills_per_iter

    def requires_trajectories(self) -> bool:
        return True

    def evolve(self, agent, target, trajectories, metrics):
        # Use both fully successful and partially successful trajectories
        successful = sorted(
            [t for t in trajectories if t.total_reward > 0],
            key=lambda t: t.total_reward, reverse=True,
        )
        logger.info("SkillEvolver: %d positive trajs out of %d", len(successful), len(trajectories))
        if not successful:
            logger.info("SkillEvolver: no successful trajectories")
            return

        skills_added = 0
        existing = {s.name for s in agent.skill_library.list_skills()}

        for traj in successful[: self._max_skills]:
            steps_text = " -> ".join(
                s.action.text.strip() for s in traj.steps if s.action.text.strip()
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract ONE reusable sub-procedure from this cooking trajectory "
                        "that could help with OTHER recipes.\n"
                        "Respond EXACTLY as:\n"
                        "SKILL_NAME: <short_name>\n"
                        "DESCRIPTION: <one line>\n"
                        "STEPS: <step1> -> <step2> -> ..."
                    ),
                },
                {"role": "user", "content": f"Recipe: {traj.task_id}\nActions: {steps_text}"},
            ]

            output = agent.brain.generate(messages, temperature=0.3, max_tokens=150)
            text = output.text.strip()

            # Parse structured response
            name = desc = steps = ""
            for line in text.split("\n"):
                low = line.strip().upper()
                if low.startswith("SKILL_NAME:"):
                    name = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                elif low.startswith("DESCRIPTION:"):
                    desc = line.split(":", 1)[1].strip()
                elif low.startswith("STEPS:"):
                    steps = line.split(":", 1)[1].strip()

            if not name or name in existing:
                continue

            try:
                skill = TextSkill(
                    name=name,
                    description=desc or f"Skill from {traj.task_id}",
                    instructions=f"Steps: {steps}" if steps else text,
                    tags=["auto_extracted", traj.task_id],
                )
                agent.skill_library.add_skill(skill)
                existing.add(name)
                skills_added += 1
                logger.info("  Extracted skill: '%s'", name)
            except Exception as e:
                logger.warning("  Failed to add skill '%s': %s", name, e)

        metrics.log({"skills/added": skills_added, "skills/total": len(agent.skill_library)})
        logger.info("SkillEvolver: +%d skills (%d total)", skills_added, len(agent.skill_library))


# ── 3. Build agent ────────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.agent.skills.library import SkillLibrary
from sea.metrics.tracker import MetricsTracker
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
            "For example, if the recipe says 'add lettuce', your action must contain 'add lettuce'."
        ),
        default_max_tokens=150,
        default_temperature=0.3,  # lower temp for more reliable keyword matching
    ),
    memory=WorkingMemory(max_size=10),
    planner=ReActPlanner(),
    skill_library=SkillLibrary(),
)

env = RecipeKitchenEnv()

# Pre-warm the skill library's embedding model (one-time cost)
logger.info("Loading embedding model for skill library ...")
try:
    agent.skill_library._ensure_loaded()
    logger.info("Embedding model loaded.")
except Exception as e:
    logger.warning("Could not load embedding model: %s (skills will use fallback)", e)

metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()
evolver = SkillExtractEvolver(max_skills_per_iter=3)

# ── 4. Baseline ───────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info(
    "Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
    baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps,
)

# ── 5. Evolution loop ─────────────────────────────────────────────────
results = [("baseline", baseline.success_rate, baseline.avg_reward, 0)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (skills: %d)",
                iteration, NUM_ITERATIONS, len(agent.skill_library))
    logger.info("=" * 60)

    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    n_ok = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", n_ok, len(trajectories))

    # Evolve skills
    skill_target = agent.evolvable_components().get("skill_library")
    if skill_target is not None:
        try:
            evolver.evolve(agent, skill_target, trajectories, metrics)
        except Exception as e:
            logger.error("Evolver failed: %s", e, exc_info=True)

    for skill in agent.skill_library.list_skills():
        logger.info("  Skill: '%s' — %s", skill.name, skill.description[:60])

    result = evaluator.evaluate(agent, [env])
    ns = len(agent.skill_library)
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward, ns))
    logger.info(
        "Iter %d: success=%.0f%%, reward=%.3f, steps=%.1f, skills=%d",
        iteration, result.success_rate * 100, result.avg_reward, result.avg_steps, ns,
    )

# ── 6. Summary ────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, sr, ar, ns in results:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f  skills=%d", name, sr * 100, ar, ns)
improvement = results[-1][1] - results[0][1]
logger.info("Improvement: %+.0f%% success rate", improvement * 100)

logger.info("Final Skill Library (%d):", len(agent.skill_library))
for s in agent.skill_library.list_skills():
    logger.info("  [%s] %s", s.name, s.description)

output_dir = Path("outputs/skill_evolution")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "summary.json").write_text(json.dumps({
    "model": MODEL,
    "results": [{"stage": n, "success_rate": s, "avg_reward": r, "skills": ns}
                for n, s, r, ns in results],
    "final_skills": [s.to_dict() for s in agent.skill_library.list_skills()],
}, indent=2, ensure_ascii=False))
logger.info("Saved to %s", output_dir / "summary.json")
