#!/usr/bin/env python3
"""Tutorial 3: SFT Evolution (LoRA Fine-Tuning) on TextCraft.

Demonstrates supervised fine-tuning of a local LLM:
- Phase A: Collect training data via API (fast, high-quality trajectories)
- Phase B: Evaluate baseline local model (Qwen2.5-7B on vLLM)
- Phase C: SFT training loop (LoRA adapter on GPU)

Expected result: success rate of local model improves significantly after
learning from API-generated demonstrations.

Usage:
    python examples/sft_textcraft/run.py

Requires:
    - GPU 4-5 for vLLM inference (TP=2)
    - GPU 6 for LoRA training
    - Qwen/Qwen2.5-7B-Instruct model downloaded
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.core.types import Trajectory
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.methods.sft import SFTEvolver
from sea.evolution.targets.lm_params import LoRATarget
from sea.metrics.evaluator import Evaluator
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INFERENCE_DEVICE = "cuda:4"  # vLLM tensor_parallel across 4,5
TRAINING_DEVICE = "cuda:6"
NUM_COLLECT = 50       # API trajectories for training data
NUM_SFT_ITERS = 3     # SFT training iterations
EVAL_EPISODES = 20
NUM_TASKS = 50
MAX_STEPS = 15
OUTPUT_DIR = Path("outputs/tutorial_sft")

env_json = Path(__file__).resolve().parent.parent.parent / "env.json"
with open(env_json) as f:
    creds = json.load(f)["aigocode-gpt"]
API_KEY = creds["apiKey"]
BASE_URL = creds["baseUrl"] + "/v1"
API_MODEL = "openai/gpt-5.4-nano"

SYSTEM_PROMPT = (
    "You are a Minecraft crafting agent. Follow recipes step by step.\n"
    "First check what you have, then gather materials, then craft.\n"
    "Use exact item names from the available actions."
)


def collect_api_data(env: TextCraftEnv, n: int) -> list[Trajectory]:
    """Collect high-quality trajectories using the API backend."""
    from sea.llm.api_backend import APIBackend

    logger.info("Phase A: Collecting %d trajectories via API...", n)
    backend = APIBackend(model=API_MODEL, base_url=BASE_URL, api_key=API_KEY)
    api_agent = SEAAgent(
        brain=LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                       default_max_tokens=150, default_temperature=0.0),
        memory=WorkingMemory(max_size=20),
        planner=ReActPlanner(),
    )
    collector = TrajectoryCollector()
    trajectories = collector.collect(api_agent, [env], n=n)
    n_success = sum(1 for t in trajectories if t.success)
    logger.info("Collected %d trajectories (%d successful, %.0f%%)",
                len(trajectories), n_success, 100 * n_success / max(len(trajectories), 1))
    return trajectories


def build_local_agent(env: TextCraftEnv) -> SEAAgent:
    """Build agent with local vLLM backend."""
    from sea.llm.vllm_backend import VLLMBackend

    logger.info("Phase B: Loading local model %s on vLLM (TP=2)...", MODEL_NAME)
    backend = VLLMBackend(
        model=MODEL_NAME,
        device=INFERENCE_DEVICE,
        tensor_parallel_size=2,
        enable_lora=True,
        max_lora_rank=32,
    )
    brain = LLMBrain(
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        default_max_tokens=150,
        default_temperature=0.0,
    )
    return SEAAgent(
        brain=brain,
        memory=WorkingMemory(max_size=20),
        planner=ReActPlanner(),
    )


def main():
    logger.info("=" * 60)
    logger.info("Tutorial 3: SFT Evolution (LoRA Fine-Tuning)")
    logger.info("=" * 60)

    env = TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)
    evaluator = Evaluator(num_episodes_per_env=EVAL_EPISODES, eval_temperature=0.0)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])

    # Phase A: Collect training data via API
    all_trajectories = collect_api_data(env, NUM_COLLECT)
    successful = [t for t in all_trajectories if t.success]
    logger.info("Using %d successful trajectories for SFT training", len(successful))

    if not successful:
        logger.error("No successful trajectories collected. Cannot proceed with SFT.")
        env.close()
        return

    # Phase B: Load local model and evaluate baseline
    agent = build_local_agent(env)
    lora_target = LoRATarget(
        base_model_name=MODEL_NAME,
        adapter_dir=OUTPUT_DIR / "adapter_init",
    )

    logger.info("Evaluating baseline (no LoRA)...")
    baseline_sr = evaluator.evaluate(agent, [env]).success_rate
    logger.info("Baseline: success_rate=%.1f%%", baseline_sr * 100)

    # Phase C: SFT training loop
    sft = SFTEvolver(
        model_name=MODEL_NAME,
        device=TRAINING_DEVICE,
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_length=1024,
        output_dir=str(OUTPUT_DIR),
        torch_dtype="bfloat16",
    )

    results_table = [("Baseline", baseline_sr)]

    for iteration in range(1, NUM_SFT_ITERS + 1):
        logger.info("\n── SFT Iteration %d/%d ──", iteration, NUM_SFT_ITERS)

        # Train on all successful trajectories (cumulative)
        sft.evolve(agent, lora_target, successful, metrics)

        # Evaluate
        sr = evaluator.evaluate(agent, [env]).success_rate
        logger.info("Iter %d: success_rate=%.1f%%", iteration, sr * 100)
        results_table.append((f"SFT Iter {iteration}", sr))

        # Collect more trajectories with the improved model for next round
        logger.info("Collecting more trajectories with improved model...")
        new_trajs = TrajectoryCollector().collect(agent, [env], n=20)
        new_success = [t for t in new_trajs if t.success]
        successful.extend(new_success)
        logger.info("Added %d new successful trajectories (total: %d)",
                    len(new_success), len(successful))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("%-15s  %-15s", "Stage", "Success Rate")
    logger.info("-" * 32)
    for stage, sr in results_table:
        logger.info("%-15s  %-15s", stage, f"{sr*100:.1f}%")

    improvement = results_table[-1][1] - results_table[0][1]
    logger.info("\nImprovement: %+.1f%%", improvement * 100)

    env.close()
    logger.info("Done! LoRA adapter saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
