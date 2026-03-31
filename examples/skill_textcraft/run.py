#!/usr/bin/env python3
"""Tutorial 2: Skill Evolution (SKILL.md) on TextCraft.

Demonstrates the SKILL.md Progressive Disclosure paradigm:
- Agent extracts reusable sub-recipes from successful TextCraft trajectories
- Skills are stored as SKILL.md files (YAML frontmatter + markdown body)
- Progressive disclosure: planner sees skill index → summaries → full content

No GPU needed — uses an API backend for LLM inference.

Expected result: success rate improves +10-15% as the agent accumulates
useful crafting skills (e.g., craft_oak_planks, make_sticks).

Usage:
    python examples/skill_textcraft/run.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.agent.skills.library import SkillLibrary
from sea.agent.skills.skill_md import SkillMd, SkillFrontmatter, parse_skill_md
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.base import Evolver
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.evaluator import Evaluator
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
NUM_ITERATIONS = 5
COLLECT_PER_ITER = 30
EVAL_EPISODES = 20
NUM_TASKS = 50
MAX_STEPS = 15
SKILLS_DIR = Path("outputs/tutorial_skills/skills")

env_json = Path(__file__).resolve().parent.parent.parent / "env.json"
with open(env_json) as f:
    creds = json.load(f)["aigocode-gpt"]
API_KEY = creds["apiKey"]
BASE_URL = creds["baseUrl"] + "/v1"
MODEL = "openai/gpt-5.4-nano"


# ── Custom Skill Extraction Evolver ─────────────────────────────────

SKILL_EXTRACT_PROMPT = """\
Analyze this successful TextCraft episode and extract ONE reusable crafting skill.

Episode actions:
{actions}

Goal: {goal}

Write the skill in SKILL.md format:
---
name: <snake_case_name>
description: <one line description>
tags: [crafting, textcraft]
when_to_use: <when this skill is useful>
---

## Steps
<numbered step-by-step instructions using exact TextCraft commands>

## Notes
<any tips or important details>

Respond with ONLY the SKILL.md content, starting with ---"""


@EVOLVER_REGISTRY.register("skill_extract_tc")
class SkillExtractEvolver(Evolver):
    """Extract reusable SKILL.md files from successful TextCraft trajectories."""

    def __init__(self, max_skills_per_iter: int = 5) -> None:
        self._max_skills = max_skills_per_iter

    def requires_trajectories(self) -> bool:
        return True

    def evolve(self, agent: SEAAgent, target, trajectories: list[Trajectory],
               metrics: MetricsTracker, **kwargs: Any) -> None:
        successes = [t for t in trajectories if t.success]
        if not successes:
            logger.info("No successful trajectories for skill extraction")
            return

        skills_added = 0
        for traj in successes[:self._max_skills]:
            # Build extraction prompt
            actions = "\n".join(f"  {i+1}. {s.action.text}" for i, s in enumerate(traj.steps))
            goal = traj.metadata.get("task_description", traj.task_id)[:200]
            prompt = SKILL_EXTRACT_PROMPT.format(actions=actions, goal=goal)

            # Ask LLM to extract skill
            output = agent.brain.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )

            # Parse SKILL.md response
            text = output.text.strip()
            if not text.startswith("---"):
                text = "---\n" + text

            try:
                skill_md = parse_skill_md(text)
            except ValueError as e:
                logger.debug("Failed to parse skill: %s", e)
                continue

            # Deduplication: skip if description is too similar
            is_dup = False
            for existing in agent.skill_library.list_skills():
                if (skill_md.description.lower() in existing.description.lower()
                        or existing.description.lower() in skill_md.description.lower()):
                    is_dup = True
                    break
            if is_dup:
                continue

            agent.skill_library.add_skill(skill_md)
            skills_added += 1
            logger.info("Extracted skill: %s", skill_md.name)

        metrics.log({
            "skills/added": skills_added,
            "skills/total": len(agent.skill_library),
        })
        logger.info("Skills: +%d new, %d total", skills_added, len(agent.skill_library))

    def save_checkpoint(self, path: Path) -> None:
        pass

    def load_checkpoint(self, path: Path) -> None:
        pass

    def state_dict(self) -> dict:
        return {}


# ── Main ────────────────────────────────────────────────────────────

def build_agent() -> SEAAgent:
    from sea.llm.api_backend import APIBackend

    backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
    brain = LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a Minecraft crafting agent. Follow recipes step by step.\n"
            "First check what you have, then gather materials, then craft.\n"
            "Use exact item names from the available actions.\n"
            "If a relevant skill is available, follow its steps."
        ),
        default_max_tokens=150,
        default_temperature=0.0,
    )
    memory = WorkingMemory(max_size=20)
    skill_library = SkillLibrary(skills_dir=SKILLS_DIR, use_embeddings=False)
    planner = ReActPlanner()
    return SEAAgent(
        brain=brain, memory=memory, planner=planner,
        skill_library=skill_library, skill_retrieval_k=3,
    )


def evaluate(agent: SEAAgent, env: TextCraftEnv, evaluator: Evaluator, label: str) -> float:
    results = evaluator.evaluate(agent, [env])
    sr = results.success_rate
    logger.info("%s: success_rate=%.1f%% (%d/%d)",
                label, sr * 100, int(sr * results.num_episodes), results.num_episodes)
    return sr


def main():
    logger.info("=" * 60)
    logger.info("Tutorial 2: Skill Evolution (SKILL.md Progressive Disclosure)")
    logger.info("=" * 60)

    agent = build_agent()
    env = TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)
    collector = TrajectoryCollector()
    evaluator = Evaluator(num_episodes_per_env=EVAL_EPISODES, eval_temperature=0.0)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])
    evolver = SkillExtractEvolver(max_skills_per_iter=5)

    # Baseline
    logger.info("\n── Baseline evaluation (no skills) ──")
    baseline_sr = evaluate(agent, env, evaluator, "Baseline")

    results_table = [("Baseline", baseline_sr, 0)]

    for iteration in range(1, NUM_ITERATIONS + 1):
        logger.info("\n── Iteration %d/%d ──", iteration, NUM_ITERATIONS)

        # Collect
        trajectories = collector.collect(agent, [env], n=COLLECT_PER_ITER)
        n_success = sum(1 for t in trajectories if t.success)
        logger.info("Collected: %d success, %d fail", n_success, len(trajectories) - n_success)

        # Evolve skills
        skill_target = agent.evolvable_components().get("skill_library")
        if skill_target is not None:
            evolver.evolve(agent, skill_target, trajectories, metrics)

        # Evaluate
        sr = evaluate(agent, env, evaluator, f"Iter {iteration}")
        results_table.append((f"Iter {iteration}", sr, len(agent.skill_library)))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s  %-15s  %-12s", "Stage", "Success Rate", "Num Skills")
    logger.info("-" * 42)
    for stage, sr, n_skills in results_table:
        logger.info("%-12s  %-15s  %-12d", stage, f"{sr*100:.1f}%", n_skills)

    improvement = results_table[-1][1] - results_table[0][1]
    logger.info("\nImprovement: %+.1f%%", improvement * 100)

    # Show extracted skills
    logger.info("\n── Extracted Skills ──")
    for skill in agent.skill_library.list_skills():
        logger.info("  - %s: %s", skill.name, skill.description)

    # Show SKILL.md files on disk
    if SKILLS_DIR.exists():
        md_files = list(SKILLS_DIR.glob("*.md"))
        logger.info("\n%d SKILL.md files saved to %s", len(md_files), SKILLS_DIR)

    env.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
