#!/usr/bin/env python3
"""ExpeL on TextCraft: evolve semantic rule memory from trajectories.

The agent collects TextCraft trajectories, distills structured rules from both
successes and failures, stores them as semantic memory, and reuses those rules
on later tasks.

Requires:
    pip install openai textcraft sentence-transformers faiss-cpu

Usage:
    export SEA_API_KEY="your-key"
    export SEA_BASE_URL="https://api.example.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/expel_textcraft/run.py
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
logger = logging.getLogger("expel_textcraft")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_KEY = require_env("SEA_API_KEY")
BASE_URL = require_env("SEA_BASE_URL")
MODEL = require_env("SEA_MODEL")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 20
OUTPUT_DIR = Path("outputs/expel_textcraft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.semantic import SemanticMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.methods.expel import ExpeLEvolver
from sea.llm.api_backend import APIBackend
from sea.metrics.evaluator import Evaluator
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.tracker import MetricsTracker

logger.info("=" * 60)
logger.info("ExpeL on TextCraft")
logger.info("=" * 60)
logger.info("Model: %s", MODEL)
logger.info("Base URL: %s", BASE_URL)

backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a Minecraft crafting agent. You are given a goal item and recipes.\n"
            "Available commands:\n"
            "- get <count> <item>\n"
            "- craft <count> <item> using <count> <ingredient>, <count> <ingredient>, ...\n"
            "- inventory\n"
            "Use relevant rules from memory when they apply. Work backwards from the goal."
        ),
        default_max_tokens=220,
        default_temperature=0.4,
    ),
    memory=SemanticMemory(max_size=1000),
    planner=ReActPlanner(),
)

env = TextCraftEnv(max_steps_val=15, num_tasks=40)
collector = TrajectoryCollector()
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
evolver = ExpeLEvolver(
    max_success_trajectories=3,
    max_failure_trajectories=3,
    max_steps_per_trajectory=8,
    max_rules_per_outcome=3,
    min_priority=0.9,
)

memory_target = agent.evolvable_components().get("memory")
if memory_target is None:
    raise RuntimeError("Agent memory is not evolvable; ExpeL requires an evolvable memory target")

logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info(
    "Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
    baseline.success_rate * 100,
    baseline.avg_reward,
    baseline.avg_steps,
)

results = [("baseline", baseline.success_rate, baseline.avg_reward, agent.memory.size())]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info(
        "Iteration %d / %d  (semantic memory: %d entries)",
        iteration,
        NUM_ITERATIONS,
        agent.memory.size(),
    )
    logger.info("=" * 60)

    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    num_success = sum(1 for traj in trajectories if traj.success)
    logger.info("Collected: %d/%d successful", num_success, len(trajectories))

    evolver.evolve(agent, memory_target, trajectories, metrics)
    logger.info("Memory after ExpeL: %d entries", agent.memory.size())

    result = evaluator.evaluate(agent, [env])
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward, agent.memory.size()))
    logger.info(
        "Iter %d: success=%.0f%%, reward=%.3f, steps=%.1f",
        iteration,
        result.success_rate * 100,
        result.avg_reward,
        result.avg_steps,
    )

logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, success_rate, avg_reward, memory_size in results:
    logger.info(
        "  %-12s success=%.0f%% reward=%.3f memory=%d",
        name,
        success_rate * 100,
        avg_reward,
        memory_size,
    )

learned_rules = [
    {
        "content": entry.content,
        "metadata": entry.metadata,
    }
    for entry in agent.memory.get_all()
    if entry.metadata.get("source") == "expel"
]

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "model": MODEL,
    "base_url": BASE_URL,
    "benchmark": "textcraft",
    "results": [
        {
            "stage": stage,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "memory_size": memory_size,
        }
        for stage, success_rate, avg_reward, memory_size in results
    ],
    "learned_rules": learned_rules,
}, indent=2))

logger.info("Saved summary to %s", OUTPUT_DIR / "summary.json")
