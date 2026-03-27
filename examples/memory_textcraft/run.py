#!/usr/bin/env python3
"""Memory Evolution on TextCraft: Reflexion-based crafting memory.

The agent attempts TextCraft crafting tasks. After each iteration,
the ICL evolver generates reflections on failed attempts (e.g.,
"I needed planks before crafting the sign") and stores exemplars
from successes. Accumulated memory helps in future episodes.

Usage:
    export SEA_API_KEY="your-key"
    export SEA_BASE_URL="https://api.example.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/memory_textcraft/run.py
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
logger = logging.getLogger("memory_textcraft")

API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 10

# ── Build components ──────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.methods.icl import ICLEvolver
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

logger.info("=" * 60)
logger.info("Memory Evolution on TextCraft")
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
        default_temperature=0.5,
    ),
    memory=EpisodicMemory(max_size=500),
    planner=ReActPlanner(),
)

env = TextCraftEnv(max_steps_val=15, num_tasks=30)
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()
evolver = ICLEvolver(
    max_reflections_per_step=5,
    max_exemplars=3,
    exemplar_selection="highest_reward",
)

# ── Baseline ──────────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ── Evolution loop ────────────────────────────────────────────────────
results = [("baseline", baseline.success_rate, baseline.avg_reward)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (memory: %d)", iteration, NUM_ITERATIONS, agent.memory.size())
    logger.info("=" * 60)

    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    n_ok = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", n_ok, len(trajectories))

    memory_target = agent.evolvable_components()["memory"]
    evolver.evolve(agent, memory_target, trajectories, metrics)
    logger.info("Memory after evolution: %d entries", agent.memory.size())

    result = evaluator.evaluate(agent, [env])
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward))
    logger.info("Iter %d: success=%.0f%%, reward=%.3f", iteration, result.success_rate * 100, result.avg_reward)

# ── Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, sr, ar in results:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f", name, sr * 100, ar)
logger.info("Improvement: %+.0f%% success rate", (results[-1][1] - results[0][1]) * 100)

output_dir = Path("outputs/memory_textcraft")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "summary.json").write_text(json.dumps({
    "model": MODEL, "benchmark": "textcraft",
    "results": [{"stage": n, "success_rate": s, "avg_reward": r} for n, s, r in results],
    "final_memory_size": agent.memory.size(),
}, indent=2))
