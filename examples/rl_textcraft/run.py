#!/usr/bin/env python3
"""RL (GRPO) Evolution on TextCraft: Environment-backed reward.

Uses GRPO with environment-in-the-loop reward: the model generates
crafting action sequences, which are parsed and executed in TextCraft
to get real rewards. This grounds the RL signal in actual task success.

Requires 2 GPUs: one for vLLM inference, one for GRPO training.

Usage:
    export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
    export SEA_INFERENCE_GPU="4"
    export SEA_TRAINING_GPU="5"
    python examples/rl_textcraft/run.py
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
logger = logging.getLogger("rl_textcraft")

MODEL_PATH = os.environ.get("SEA_MODEL_PATH", "/root/models/Qwen3.5-9B")
INFERENCE_GPU = os.environ.get("SEA_INFERENCE_GPU", "4")
TRAINING_GPU = os.environ.get("SEA_TRAINING_GPU", "5")

NUM_ITERATIONS = 3
NUM_COLLECT = 20
NUM_EVAL = 10
OUTPUT_DIR = Path("outputs/rl_textcraft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{INFERENCE_GPU},{TRAINING_GPU}"

# ── Build components ──────────────────────────────────────────────────
from sea.llm.vllm_backend import VLLMBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.methods.rl import RLEvolver
from sea.evolution.targets.lm_params import LoRATarget
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

logger.info("=" * 60)
logger.info("RL (GRPO) Evolution on TextCraft")
logger.info("=" * 60)

backend = VLLMBackend(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
    device="cuda:0",
    enforce_eager=True,
)

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a Minecraft crafting agent. You receive crafting recipes and a goal.\n"
            "Commands: get <count> <item> | craft <count> <item> using <ingredients> | inventory\n"
            "Work backwards from the goal. Get raw materials first, then craft.\n"
            "Respond with:\nThought: <reasoning>\nAction: <command>"
        ),
        default_max_tokens=256,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=200),
    planner=ReActPlanner(),
)

env = TextCraftEnv(max_steps_val=15, num_tasks=50)

# Create a separate env instance for GRPO reward evaluation
reward_env = TextCraftEnv(max_steps_val=15, num_tasks=50)

# LoRA target for RL
lora_target = LoRATarget(
    base_model_name=MODEL_PATH,
    adapter_dir=OUTPUT_DIR / "adapter_init",
)

# GRPO evolver with environment-backed reward
evolver = RLEvolver(
    model_name=MODEL_PATH,
    algorithm="grpo",
    device="cuda:1",
    learning_rate=1e-5,
    num_epochs=1,
    batch_size=2,
    gradient_accumulation_steps=4,
    max_completion_length=512,
    num_generations=4,
    output_dir=str(OUTPUT_DIR),
    envs=[reward_env],  # Environment for reward computation
)

metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()

# ── Baseline ──────────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ── Evolution loop ────────────────────────────────────────────────────
results = [("baseline", baseline.success_rate, baseline.avg_reward)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d", iteration, NUM_ITERATIONS)
    logger.info("=" * 60)

    # Collect trajectories for prompts
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    n_ok = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", n_ok, len(trajectories))

    # GRPO evolution with env-backed reward
    evolver.evolve(agent, lora_target, trajectories, metrics, envs=[reward_env])

    # Hot-swap new adapter
    adapter_path = lora_target.get_evolvable_state()
    if adapter_path.exists():
        agent.brain.backend.load_lora(str(adapter_path), name=f"iter_{iteration}")
        agent.brain.lora_name = f"iter_{iteration}"
        logger.info("LoRA hot-swapped: %s", adapter_path)

    # Evaluate
    result = evaluator.evaluate(agent, [env])
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward))
    logger.info("Iter %d: success=%.0f%%, reward=%.3f (baseline %.0f%%)",
                iteration, result.success_rate * 100, result.avg_reward, baseline.success_rate * 100)

# ── Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, sr, ar in results:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f", name, sr * 100, ar)
logger.info("Improvement: %+.0f%% success rate", (results[-1][1] - results[0][1]) * 100)

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "model": MODEL_PATH, "benchmark": "textcraft", "algorithm": "grpo",
    "results": [{"stage": n, "success_rate": s, "avg_reward": r} for n, s, r in results],
}, indent=2))
