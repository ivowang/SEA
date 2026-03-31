#!/usr/bin/env python3
"""Tutorial 1: Memory Evolution (ICL/Reflexion) on TextCraft.

Demonstrates how the SEA agent evolves its working memory through
reflections on failures and exemplars from successes. No GPU needed —
uses an API backend for LLM inference.

Expected result: success rate improves +10-20% over 5 iterations as
the agent accumulates useful reflections and exemplars in memory.

Usage:
    python examples/memory_textcraft/run.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.methods.icl import ICLEvolver
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
MAX_STEPS = 30

# Load API credentials from env.json
env_json = Path(__file__).resolve().parent.parent.parent / "env.json"
with open(env_json) as f:
    creds = json.load(f)["aigocode-gpt"]
API_KEY = creds["apiKey"]
BASE_URL = creds["baseUrl"] + "/v1"
MODEL = "openai/gpt-5.4-nano"


def build_agent() -> SEAAgent:
    """Build an agent with API backend and working memory."""
    from sea.llm.api_backend import APIBackend

    backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
    brain = LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a TextCraft crafting agent. You solve crafting goals one step at a time.\n\n"
            "RULES:\n"
            "- The observation shows available crafting recipes and your goal\n"
            "- Execute ONE command per turn\n"
            "- Commands: 'get <count> <item>' for base materials, 'craft <recipe>' following the exact recipe shown\n"
            "- Work bottom-up: get base materials first, craft intermediate items, then the final goal\n"
            "- Copy recipes EXACTLY as shown (including counts). Do not modify the recipe.\n"
            "- NEVER use finish(). The environment ends when the goal is crafted.\n\n"
            "EXAMPLE (goal: craft 4 oak planks):\n"
            "Thought: The recipe says 'craft 4 oak planks using 1 oak logs'. I need 1 oak logs first.\n"
            "Action: get 1 oak logs"
        ),
        default_max_tokens=150,
        default_temperature=0.0,
    )
    memory = WorkingMemory(max_size=50)
    planner = ReActPlanner()
    return SEAAgent(brain=brain, memory=memory, planner=planner)


def evaluate(agent: SEAAgent, env: TextCraftEnv, evaluator: Evaluator, label: str) -> float:
    """Run evaluation and return success rate."""
    results = evaluator.evaluate(agent, [env])
    sr = results.success_rate
    logger.info("%s: success_rate=%.1f%% (%d/%d episodes)",
                label, sr * 100, int(sr * results.num_episodes), results.num_episodes)
    return sr


def main():
    logger.info("=" * 60)
    logger.info("Tutorial 1: Memory Evolution (ICL/Reflexion)")
    logger.info("=" * 60)

    # ── Setup ──
    agent = build_agent()
    env = TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)
    collector = TrajectoryCollector()
    evaluator = Evaluator(num_episodes_per_env=EVAL_EPISODES, eval_temperature=0.0)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])
    evolver = ICLEvolver(
        max_reflections_per_step=5,
        max_exemplars=10,
        exemplar_selection="diverse",
    )

    # ── Baseline ──
    logger.info("\n── Baseline evaluation ──")
    baseline_sr = evaluate(agent, env, evaluator, "Baseline")

    # ── Evolution loop ──
    results_table = [("Baseline", baseline_sr, 0)]

    for iteration in range(1, NUM_ITERATIONS + 1):
        logger.info("\n── Iteration %d/%d ──", iteration, NUM_ITERATIONS)

        # Collect trajectories
        logger.info("Collecting %d trajectories...", COLLECT_PER_ITER)
        trajectories = collector.collect(agent, [env], n=COLLECT_PER_ITER)
        n_success = sum(1 for t in trajectories if t.success)
        n_fail = len(trajectories) - n_success
        logger.info("Collected: %d success, %d fail", n_success, n_fail)

        # Evolve memory
        memory_target = agent.evolvable_components()["memory"]
        evolver.evolve(agent, memory_target, trajectories, metrics)
        logger.info("Memory size after evolution: %d entries", agent.memory.size())

        # Evaluate
        sr = evaluate(agent, env, evaluator, f"Iter {iteration}")
        results_table.append((f"Iter {iteration}", sr, agent.memory.size()))

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s  %-15s  %-12s", "Stage", "Success Rate", "Memory Size")
    logger.info("-" * 42)
    for stage, sr, mem_size in results_table:
        logger.info("%-12s  %-15s  %-12d", stage, f"{sr*100:.1f}%", mem_size)

    improvement = results_table[-1][1] - results_table[0][1]
    logger.info("\nImprovement: %+.1f%% (%.1f%% → %.1f%%)",
                improvement * 100,
                results_table[0][1] * 100,
                results_table[-1][1] * 100)

    env.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
