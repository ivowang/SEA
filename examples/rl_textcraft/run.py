#!/usr/bin/env python3
"""Tutorial 4: RL Evolution (REINFORCE) on TextCraft.

Demonstrates offline trajectory-level REINFORCE:
- Phase A: Collect trajectories via API with high concurrency (no GPU)
- Phase B: Train/eval loop on a single GPU — alternating between
  REINFORCE training (HFTrainingBackend) and inference (vLLM)

Unlike SFT (which only learns from successes), REINFORCE learns from
BOTH successes and failures via advantage-weighted policy gradient.

Only 1 GPU needed. Training and inference share the same card.

Usage:
    python examples/rl_textcraft/run.py

Requires:
    - 1 GPU (default: cuda:7) with ~30GB free
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
from sea.evolution.methods.rl import RLEvolver
from sea.evolution.targets.lm_params import LoRATarget
from sea.metrics.evaluator import Evaluator
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_DEVICE = "cuda:7"          # single GPU for both training and inference
NUM_COLLECT = 80               # need both successes AND failures
NUM_RL_ITERS = 3
EVAL_EPISODES = 20
NUM_TASKS = 50
MAX_STEPS = 30
OUTPUT_DIR = Path("outputs/tutorial_rl")

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
    """Collect trajectories via API with high concurrency (no GPU needed).

    Both successes and failures are needed for REINFORCE.
    """
    from sea.llm.api_backend import APIBackend

    logger.info("Phase A: Collecting %d trajectories via API (%d concurrent)...", n, max_workers)

    def make_agent():
        backend = APIBackend(model=API_MODEL, base_url=BASE_URL, api_key=API_KEY)
        return SEAAgent(
            brain=LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                           default_max_tokens=150, default_temperature=0.3),
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
    n_fail = len(trajectories) - n_success
    logger.info("Collected: %d success, %d fail (both needed for REINFORCE)", n_success, n_fail)
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
    logger.info("Tutorial 4: RL Evolution (REINFORCE)")
    logger.info("Single GPU mode: %s", GPU_DEVICE)
    logger.info("=" * 60)

    env = TextCraftEnv(max_steps_val=MAX_STEPS, num_tasks=NUM_TASKS)
    evaluator = Evaluator(num_episodes_per_env=EVAL_EPISODES, eval_temperature=0.0)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])

    # ── Phase A: Collect training data via API ──
    all_trajectories = collect_api_data(NUM_COLLECT, max_workers=30)
    n_success = sum(1 for t in all_trajectories if t.success)
    avg_reward = sum(t.total_reward for t in all_trajectories) / max(len(all_trajectories), 1)
    logger.info("Training data: %d trajectories (%.0f%% success, avg_reward=%.2f)",
                len(all_trajectories), 100 * n_success / max(len(all_trajectories), 1), avg_reward)

    lora_target = LoRATarget(
        base_model_name=MODEL_NAME,
        adapter_dir=OUTPUT_DIR / "adapter_init",
    )

    rl = RLEvolver(
        model_name=MODEL_NAME,
        algorithm="reinforce",
        device=GPU_DEVICE,
        learning_rate=1e-5,
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=1024,
        gamma=0.99,
        entropy_coeff=0.01,
        output_dir=str(OUTPUT_DIR),
        torch_dtype="bfloat16",
    )

    results_table = []

    # ── Phase B: Train/Eval loop (single GPU, alternating) ──
    for iteration in range(NUM_RL_ITERS + 1):
        # --- Eval stage: load vLLM on GPU ---
        lora_path = None
        if iteration > 0:
            state = lora_target.get_evolvable_state()
            lora_path = str(state) if state and Path(str(state)).exists() else None

        label = "Baseline" if iteration == 0 else f"RL Iter {iteration}"
        logger.info("\n── %s: Loading vLLM for evaluation ──", label)
        agent = load_vllm_agent(lora_path=lora_path)

        sr = evaluator.evaluate(agent, [env]).success_rate
        logger.info("%s: success_rate=%.1f%%", label, sr * 100)
        results_table.append((label, sr))

        # Shutdown vLLM to free GPU for training
        shutdown_vllm(agent)

        if iteration == NUM_RL_ITERS:
            break

        # --- Train stage: REINFORCE on same GPU ---
        logger.info("\n── REINFORCE Training %d/%d ──", iteration + 1, NUM_RL_ITERS)
        from unittest.mock import MagicMock
        dummy_agent = MagicMock()
        dummy_agent.brain = MagicMock()
        dummy_agent.brain.swap_lora = MagicMock()
        dummy_agent.brain.system_prompt = SYSTEM_PROMPT

        rl.evolve(dummy_agent, lora_target, all_trajectories, metrics)
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
    logger.info("\nKey difference from SFT:")
    logger.info("  SFT only learns from successes (imitation learning)")
    logger.info("  REINFORCE learns from both successes AND failures (policy gradient)")

    env.close()
    logger.info("Done! LoRA adapter saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
