#!/usr/bin/env python3
"""Skill Evolution on TextCraft: Extracting reusable sub-recipes.

The agent attempts TextCraft crafting tasks. A custom SkillExtractEvolver
identifies reusable sub-recipes from successful trajectories (e.g.,
"make_planks: get 1 oak log → craft 4 oak planks"). Skills are stored
in the SkillLibrary and retrieved for new tasks sharing sub-recipes.

Usage:
    export SEA_API_KEY="your-key"
    export SEA_BASE_URL="https://api.example.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/skill_textcraft/run.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("skill_textcraft")

API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 10

# ── Custom SkillExtractEvolver ────────────────────────────────────────
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.base import Evolvable
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.agent.skills.code_skill import TextSkill
from sea.metrics.tracker import MetricsTracker as _MT


# Avoid duplicate registration across runs
if "skill_extract_tc" not in EVOLVER_REGISTRY:

    @EVOLVER_REGISTRY.register("skill_extract_tc")
    class SkillExtractEvolver(Evolver):
        """Extracts reusable crafting sub-recipes from TextCraft trajectories."""

        def __init__(self, max_skills_per_iter: int = 3):
            self._max_skills = max_skills_per_iter

        def requires_trajectories(self) -> bool:
            return True

        def evolve(self, agent, target, trajectories, metrics, **kwargs):
            positive = sorted(
                [t for t in trajectories if t.total_reward > 0],
                key=lambda t: t.total_reward, reverse=True,
            )
            if not positive:
                logger.info("SkillEvolver: no positive trajectories")
                return

            existing = {s.name for s in agent.skill_library.list_skills()}
            skills_added = 0

            for traj in positive[:self._max_skills]:
                steps_text = " → ".join(s.action.text.strip() for s in traj.steps if s.action.text.strip())
                messages = [
                    {"role": "system", "content": (
                        "You are analyzing a Minecraft crafting trajectory. "
                        "Extract ONE reusable sub-recipe that could help craft OTHER items.\n"
                        "Focus on intermediate steps like making planks, sticks, dyes, etc.\n"
                        "Respond EXACTLY as:\n"
                        "SKILL_NAME: <short_name>\n"
                        "DESCRIPTION: <what this sub-recipe produces>\n"
                        "STEPS: <step1> → <step2> → ..."
                    )},
                    {"role": "user", "content": f"Task: {traj.task_id}\nGoal: {traj.metadata.get('task_description','')}\nActions: {steps_text}"},
                ]

                output = agent.brain.generate(messages, temperature=0.3, max_tokens=150)
                text = output.text.strip()

                name = desc = steps = ""
                for line in text.split("\n"):
                    up = line.strip().upper()
                    if up.startswith("SKILL_NAME:"):
                        name = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                    elif up.startswith("DESCRIPTION:"):
                        desc = line.split(":", 1)[1].strip()
                    elif up.startswith("STEPS:"):
                        steps = line.split(":", 1)[1].strip()

                if not name or name in existing:
                    continue

                try:
                    skill = TextSkill(
                        name=name,
                        description=desc or f"Sub-recipe from {traj.task_id}",
                        instructions=f"Steps: {steps}" if steps else text,
                        tags=["auto_extracted", traj.task_id],
                    )
                    agent.skill_library.add_skill(skill)
                    existing.add(name)
                    skills_added += 1
                    logger.info("  Extracted skill: '%s' — %s", name, desc[:60])
                except Exception as e:
                    logger.warning("  Failed to add skill: %s", e)

            metrics.log({"skills/added": skills_added, "skills/total": len(agent.skill_library)})
            logger.info("SkillEvolver: +%d skills (%d total)", skills_added, len(agent.skill_library))
else:
    SkillExtractEvolver = EVOLVER_REGISTRY["skill_extract_tc"]


# ── Build components ──────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.agent.skills.library import SkillLibrary
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

logger.info("=" * 60)
logger.info("Skill Evolution on TextCraft")
logger.info("=" * 60)

backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a Minecraft crafting agent. You are given a goal item to craft "
            "and a list of available crafting recipes.\n"
            "Available commands:\n"
            "- get <count> <item>: obtain raw materials\n"
            "- craft <count> <item> using <count> <ingredient>, <count> <ingredient>, ...\n"
            "- inventory: check what you have\n"
            "Plan your crafting steps carefully. Work backwards from the goal."
        ),
        default_max_tokens=200,
        default_temperature=0.3,
    ),
    memory=WorkingMemory(max_size=10),
    planner=ReActPlanner(),
    skill_library=SkillLibrary(),
)

# Pre-warm embedding model
logger.info("Loading embedding model ...")
try:
    agent.skill_library._ensure_loaded()
except Exception as e:
    logger.warning("Embedding model load failed: %s", e)

env = TextCraftEnv(max_steps_val=15, num_tasks=30)
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()
evolver = SkillExtractEvolver(max_skills_per_iter=3)

# ── Baseline ──────────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f", baseline.success_rate * 100, baseline.avg_reward)

# ── Evolution loop ────────────────────────────────────────────────────
results = [("baseline", baseline.success_rate, baseline.avg_reward, 0)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (skills: %d)", iteration, NUM_ITERATIONS, len(agent.skill_library))
    logger.info("=" * 60)

    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    n_ok = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", n_ok, len(trajectories))

    skill_target = agent.evolvable_components().get("skill_library")
    if skill_target is not None:
        evolver.evolve(agent, skill_target, trajectories, metrics)

    for s in agent.skill_library.list_skills():
        logger.info("  Skill: '%s' — %s", s.name, s.description[:60])

    result = evaluator.evaluate(agent, [env])
    ns = len(agent.skill_library)
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward, ns))
    logger.info("Iter %d: success=%.0f%%, reward=%.3f, skills=%d",
                iteration, result.success_rate * 100, result.avg_reward, ns)

# ── Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, sr, ar, ns in results:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f  skills=%d", name, sr * 100, ar, ns)
logger.info("Improvement: %+.0f%% success rate", (results[-1][1] - results[0][1]) * 100)

output_dir = Path("outputs/skill_textcraft")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "summary.json").write_text(json.dumps({
    "model": MODEL, "benchmark": "textcraft",
    "results": [{"stage": n, "success_rate": s, "avg_reward": r, "skills": ns}
                for n, s, r, ns in results],
    "final_skills": [s.to_dict() for s in agent.skill_library.list_skills()],
}, indent=2, ensure_ascii=False))
