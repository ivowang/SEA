#!/usr/bin/env python3
"""Tutorial 3: SFT Evolution (LoRA Fine-Tuning) on TextCraft.

Demonstrates supervised fine-tuning of a local LLM:
- Phase A: Collect training data via API with high concurrency (no GPU)
- Phase B: Train/eval loop on a single GPU — alternating between
  LoRA training (HFTrainingBackend) and inference (vLLM)

Only 1 GPU needed. Training and inference share the same card by
loading/unloading models between stages.

Usage:
    python examples/sft_textcraft/run.py

Requires:
    - 1 GPU (default: cuda:6) with ~30GB free
    - Qwen/Qwen2.5-7B-Instruct model downloaded
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

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
GPU_DEVICE = "cuda:6"          # single GPU for both training and inference
NUM_COLLECT = 50               # API trajectories for training data
NUM_SFT_ITERS = 3
EVAL_EPISODES = 20
NUM_TASKS = 50
MAX_STEPS = 30
OUTPUT_DIR = Path("outputs/tutorial_sft")

env_json = Path(__file__).resolve().parent.parent.parent / "env.json"
with open(env_json) as f:
    creds = json.load(f)["aigocode-gpt"]
API_KEY = creds["apiKey"]
BASE_URL = creds["baseUrl"] + "/v1"
API_MODEL = "openai/gpt-5.4-nano"

SYSTEM_PROMPT = (
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
)


def collect_api_data(n: int, max_workers: int = 30) -> list[Trajectory]:
    """Collect trajectories via API with high concurrency (no GPU needed)."""
    from sea.llm.api_backend import APIBackend

    logger.info("Phase A: Collecting %d trajectories via API (%d concurrent)...", n, max_workers)

    def make_agent():
        backend = APIBackend(model=API_MODEL, base_url=BASE_URL, api_key=API_KEY)
        return SEAAgent(
            brain=LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                           default_max_tokens=150, default_temperature=0.0),
            memory=WorkingMemory(max_size=20),
            planner=ReActPlanner(),
        )

    def make_env():
        return TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)

    trajectories = TrajectoryCollector.collect_parallel(
        agent_factory=make_agent,
        env_factory=make_env,
        n=n,
        max_workers=max_workers,
    )
    n_success = sum(1 for t in trajectories if t.success)
    logger.info("Collected %d trajectories (%d successful, %.0f%%)",
                len(trajectories), n_success, 100 * n_success / max(len(trajectories), 1))
    return trajectories


def load_vllm_agent(lora_path: str | None = None) -> SEAAgent:
    """Load vLLM on the single GPU for evaluation."""
    from sea.llm.vllm_backend import VLLMBackend

    backend = VLLMBackend(
        model=MODEL_NAME,
        device=GPU_DEVICE,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=32,
    )
    brain = LLMBrain(
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        default_max_tokens=150,
        default_temperature=0.0,
        lora_path=lora_path,
    )
    return SEAAgent(
        brain=brain,
        memory=WorkingMemory(max_size=20),
        planner=ReActPlanner(),
    )


def shutdown_vllm(agent: SEAAgent) -> None:
    """Shutdown vLLM and free GPU memory."""
    try:
        if hasattr(agent.brain.backend, 'llm'):
            from vllm.distributed.parallel_state import destroy_model_parallel
            del agent.brain.backend.llm
            destroy_model_parallel()
    except Exception:
        pass
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("vLLM shutdown, GPU memory freed")


def main():
    logger.info("=" * 60)
    logger.info("Tutorial 3: SFT Evolution (LoRA Fine-Tuning)")
    logger.info("Single GPU mode: %s", GPU_DEVICE)
    logger.info("=" * 60)

    env = TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)
    evaluator = Evaluator(num_episodes_per_env=EVAL_EPISODES, eval_temperature=0.0)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])

    # ── Phase A: Collect training data via API ──
    all_trajectories = collect_api_data(NUM_COLLECT, max_workers=30)
    successful = [t for t in all_trajectories if t.success]
    logger.info("Using %d successful trajectories for SFT training", len(successful))

    if not successful:
        logger.error("No successful trajectories collected. Cannot proceed.")
        env.close()
        return

    lora_target = LoRATarget(
        base_model_name=MODEL_NAME,
        adapter_dir=OUTPUT_DIR / "adapter_init",
    )

    sft = SFTEvolver(
        model_name=MODEL_NAME,
        device=GPU_DEVICE,
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_length=1024,
        output_dir=str(OUTPUT_DIR),
        torch_dtype="bfloat16",
    )

    results_table = []

    # ── Phase B: Train/Eval loop (single GPU, alternating) ──
    for iteration in range(NUM_SFT_ITERS + 1):
        # --- Eval stage: load vLLM on GPU ---
        lora_path = None
        if iteration > 0:
            state = lora_target.get_evolvable_state()
            lora_path = str(state) if state and Path(str(state)).exists() else None

        label = "Baseline" if iteration == 0 else f"SFT Iter {iteration}"
        logger.info("\n── %s: Loading vLLM for evaluation ──", label)
        agent = load_vllm_agent(lora_path=lora_path)

        sr = evaluator.evaluate(agent, [env]).success_rate
        logger.info("%s: success_rate=%.1f%%", label, sr * 100)
        results_table.append((label, sr))

        # Shutdown vLLM to free GPU for training
        shutdown_vllm(agent)

        if iteration == NUM_SFT_ITERS:
            break  # done, no more training

        # --- Train stage: load HF model on same GPU ---
        logger.info("\n── SFT Training %d/%d ──", iteration + 1, NUM_SFT_ITERS)
        # SFTEvolver.evolve() loads model internally, trains, saves, frees
        # We pass a dummy agent just for brain.swap_lora (which will be a no-op
        # since vLLM is down; the adapter is saved to lora_target)
        from unittest.mock import MagicMock
        dummy_agent = MagicMock()
        dummy_agent.brain = MagicMock()
        dummy_agent.brain.swap_lora = MagicMock()  # skip hot-swap (no vLLM running)
        dummy_agent.brain.system_prompt = SYSTEM_PROMPT

        sft.evolve(dummy_agent, lora_target, successful, metrics)
        logger.info("Adapter saved. Freeing training GPU memory...")

    # ── Summary ──
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
