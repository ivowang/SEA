#!/usr/bin/env python3
"""Dynamic Skill Composition on ALFWorld.

Phase 1: Extract atomic skills from successful ALFWorld trajectories
Phase 2: Compose atomic skills into higher-level compound skills using a strong LLM
Phase 3: Evaluate with composed skills vs atomic-only vs no skills

Uses gpt-5.4-nano for agent inference, gpt-5.4 for skill composition.

Usage:
    export SEA_API_KEY="your-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"
    python ideas/skill-composition/run.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("skill_composition")

API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
AGENT_MODEL = "openai/gpt-5.4-nano"      # cheap model for agent inference
COMPOSER_MODEL = "openai/gpt-5.4"        # strong model for skill composition

NUM_COLLECT = 20  # trajectories for skill extraction
NUM_EVAL = 15
OUTPUT_DIR = Path("ideas/skill-composition/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build components ──────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.agent.skills.library import SkillLibrary
from sea.agent.skills.code_skill import TextSkill, ComposedSkill
from sea.env.benchmarks.alfworld import ALFWorldEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator
from sea.core.types import Trajectory

logger.info("=" * 60)
logger.info("Dynamic Skill Composition on ALFWorld")
logger.info("=" * 60)

agent_backend = APIBackend(model=AGENT_MODEL, base_url=BASE_URL, api_key=API_KEY)
composer_backend = APIBackend(model=COMPOSER_MODEL, base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a household robot agent in ALFWorld. "
    "You must complete tasks like picking up objects, cleaning them, heating them, etc. "
    "Choose actions from the available actions list. "
    "Think step by step about what to do next."
)


def make_agent(skill_library=None):
    return SEAAgent(
        brain=LLMBrain(
            backend=agent_backend,
            system_prompt=SYSTEM_PROMPT,
            default_max_tokens=150,
            default_temperature=0.0,
        ),
        memory=WorkingMemory(max_size=5),
        planner=ReActPlanner(),
        skill_library=skill_library or SkillLibrary(),
    )


env = ALFWorldEnv(split="eval_out_of_distribution", max_steps_val=30)
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0, eval_seed=42)
collector = TrajectoryCollector()

# ══════════════════════════════════════════════════════════════════════
# Phase 0: Baseline (no skills)
# ══════════════════════════════════════════════════════════════════════
logger.info("Phase 0: Baseline evaluation (no skills)")
agent_baseline = make_agent()
baseline = evaluator.evaluate(agent_baseline, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Collect trajectories and extract atomic skills
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("Phase 1: Collecting trajectories and extracting atomic skills")
logger.info("=" * 60)

agent_collect = make_agent()
trajectories = collector.collect(agent_collect, [env], n=NUM_COLLECT)
successful = [t for t in trajectories if t.success]
logger.info("Collected %d trajectories, %d successful", len(trajectories), len(successful))

# Extract atomic skills from successful trajectories
atomic_actions = []
for traj in successful:
    for step in traj.steps:
        action = step.action.text.strip()
        if action:
            atomic_actions.append(action)

# Find common action patterns
action_counts = Counter(atomic_actions)
logger.info("Top 10 actions: %s", action_counts.most_common(10))

# Ask LLM to identify atomic skill categories
if successful:
    all_actions = "\n".join(f"- {a}" for a, c in action_counts.most_common(30))
    messages = [
        {"role": "system", "content": (
            "You are analyzing successful household robot trajectories. "
            "Identify 5-8 ATOMIC SKILLS (reusable action patterns). "
            "For each skill, provide:\n"
            "SKILL: <name>\nDESCRIPTION: <what it does>\nACTIONS: <typical action commands>"
        )},
        {"role": "user", "content": f"Common actions from successful trajectories:\n{all_actions}"},
    ]
    output = LLMBrain(backend=composer_backend, default_max_tokens=500).generate(messages, temperature=0.3)
    logger.info("Atomic skills identified:\n%s", output.text[:500])

    # Parse atomic skills
    atomic_skills = []
    current_name = current_desc = current_actions = ""
    for line in output.text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SKILL:"):
            if current_name:
                skill = TextSkill(
                    name=current_name.lower().replace(" ", "_"),
                    description=current_desc,
                    instructions=f"Actions: {current_actions}",
                    tags=["atomic"],
                )
                atomic_skills.append(skill)
            current_name = line.split(":", 1)[1].strip()
            current_desc = current_actions = ""
        elif line.upper().startswith("DESCRIPTION:"):
            current_desc = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ACTIONS:"):
            current_actions = line.split(":", 1)[1].strip()
    if current_name:
        atomic_skills.append(TextSkill(
            name=current_name.lower().replace(" ", "_"),
            description=current_desc,
            instructions=f"Actions: {current_actions}",
            tags=["atomic"],
        ))

    logger.info("Extracted %d atomic skills", len(atomic_skills))
    for s in atomic_skills:
        logger.info("  [atomic] %s: %s", s.name, s.description[:60])

# ══════════════════════════════════════════════════════════════════════
# Phase 1.5: Evaluate with atomic skills only
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("Phase 1.5: Evaluate with atomic skills")
logger.info("=" * 60)

atomic_lib = SkillLibrary()
atomic_lib._ensure_loaded()
for s in atomic_skills:
    atomic_lib.add_skill(s)

agent_atomic = make_agent(skill_library=atomic_lib)
result_atomic = evaluator.evaluate(agent_atomic, [env])
logger.info("With atomic skills: success=%.0f%%, reward=%.3f",
            result_atomic.success_rate * 100, result_atomic.avg_reward)

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Compose atomic skills into higher-level skills
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("Phase 2: Composing higher-level skills (using %s)", COMPOSER_MODEL)
logger.info("=" * 60)

# Analyze successful trajectory patterns to find composable sequences
if successful:
    traj_patterns = []
    for traj in successful[:10]:
        task_type = traj.task_type or "unknown"
        actions = [s.action.text.strip() for s in traj.steps if s.action.text.strip()]
        traj_patterns.append(f"[{task_type}] " + " → ".join(actions))

    patterns_text = "\n".join(traj_patterns)
    atomic_names = [s.name for s in atomic_skills]

    messages = [
        {"role": "system", "content": (
            "You are composing higher-level skills from atomic skills for a household robot.\n"
            f"Available atomic skills: {atomic_names}\n\n"
            "Analyze the trajectory patterns and create 3-5 COMPOSED SKILLS "
            "that combine frequently co-occurring atomic skills.\n"
            "For each composed skill:\n"
            "COMPOSED_SKILL: <name>\n"
            "DESCRIPTION: <what task it accomplishes>\n"
            "SUB_SKILLS: <skill1>, <skill2>, ...\n"
            "PLAN: <step-by-step natural language plan>"
        )},
        {"role": "user", "content": f"Successful trajectory patterns:\n{patterns_text}"},
    ]
    output = LLMBrain(backend=composer_backend, default_max_tokens=800).generate(messages, temperature=0.3)
    logger.info("Composed skills:\n%s", output.text[:800])

    # Parse composed skills
    composed_skills = []
    current = {}
    for line in output.text.split("\n"):
        line = line.strip()
        if line.upper().startswith("COMPOSED_SKILL:"):
            if current.get("name"):
                cs = ComposedSkill(
                    name=current["name"].lower().replace(" ", "_"),
                    description=current.get("desc", ""),
                    composition_plan=current.get("plan", ""),
                    sub_skills=[s.strip() for s in current.get("subs", "").split(",")],
                    tags=["composed"],
                )
                composed_skills.append(cs)
            current = {"name": line.split(":", 1)[1].strip()}
        elif line.upper().startswith("DESCRIPTION:"):
            current["desc"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUB_SKILLS:"):
            current["subs"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PLAN:"):
            current["plan"] = line.split(":", 1)[1].strip()
    if current.get("name"):
        composed_skills.append(ComposedSkill(
            name=current["name"].lower().replace(" ", "_"),
            description=current.get("desc", ""),
            composition_plan=current.get("plan", ""),
            sub_skills=[s.strip() for s in current.get("subs", "").split(",")],
            tags=["composed"],
        ))

    logger.info("Composed %d higher-level skills", len(composed_skills))
    for s in composed_skills:
        logger.info("  [composed] %s: %s (subs: %s)", s.name, s.description[:50], s.info.sub_skills)

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Evaluate with composed skills
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("Phase 3: Evaluate with atomic + composed skills")
logger.info("=" * 60)

composed_lib = SkillLibrary()
composed_lib._ensure_loaded()
for s in atomic_skills:
    composed_lib.add_skill(s)
for s in composed_skills:
    composed_lib.add_skill(s)

agent_composed = make_agent(skill_library=composed_lib)
result_composed = evaluator.evaluate(agent_composed, [env])
logger.info("With composed skills: success=%.0f%%, reward=%.3f",
            result_composed.success_rate * 100, result_composed.avg_reward)

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("RESULTS SUMMARY")
logger.info("=" * 60)
logger.info("  No skills:      success=%.0f%%  reward=%.3f",
            baseline.success_rate * 100, baseline.avg_reward)
logger.info("  Atomic only:    success=%.0f%%  reward=%.3f  skills=%d",
            result_atomic.success_rate * 100, result_atomic.avg_reward, len(atomic_skills))
logger.info("  Atomic+Composed: success=%.0f%%  reward=%.3f  skills=%d",
            result_composed.success_rate * 100, result_composed.avg_reward,
            len(atomic_skills) + len(composed_skills))

summary = {
    "benchmark": "alfworld",
    "agent_model": AGENT_MODEL,
    "composer_model": COMPOSER_MODEL,
    "results": {
        "baseline": {"success_rate": baseline.success_rate, "avg_reward": baseline.avg_reward},
        "atomic_only": {"success_rate": result_atomic.success_rate, "avg_reward": result_atomic.avg_reward,
                        "num_skills": len(atomic_skills)},
        "composed": {"success_rate": result_composed.success_rate, "avg_reward": result_composed.avg_reward,
                     "num_skills": len(atomic_skills) + len(composed_skills)},
    },
    "atomic_skills": [s.to_dict() for s in atomic_skills],
    "composed_skills": [s.to_dict() for s in composed_skills],
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
logger.info("Saved to %s", OUTPUT_DIR / "summary.json")
