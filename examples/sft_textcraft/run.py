#!/usr/bin/env python3
"""SFT Evolution on TextCraft: LoRA fine-tuning on successful crafting.

Collects crafting trajectories on TextCraft with a local LLM (vLLM),
filters successful ones, trains a LoRA adapter via SFT, hot-swaps
into vLLM, and evaluates improvement.

Requires 2 GPUs: one for vLLM inference, one for SFT training.

Usage:
    export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
    export SEA_INFERENCE_GPU="4"
    export SEA_TRAINING_GPU="5"
    python examples/sft_textcraft/run.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("sft_textcraft")

MODEL_PATH = os.environ.get("SEA_MODEL_PATH", "/root/models/Qwen3.5-9B")
INFERENCE_GPU = os.environ.get("SEA_INFERENCE_GPU", "4")
TRAINING_GPU = os.environ.get("SEA_TRAINING_GPU", "5")

NUM_ITERATIONS = 3
NUM_COLLECT = 20
NUM_EVAL = 10
OUTPUT_DIR = Path("outputs/sft_textcraft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = f"{INFERENCE_GPU},{TRAINING_GPU}"

# ── Load vLLM ─────────────────────────────────────────────────────────
from sea.llm.vllm_backend import VLLMBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.data.dataset import trajectories_to_sft_data, to_hf_dataset
from sea.llm.hf_backend import HFTrainingBackend
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

logger.info("=" * 60)
logger.info("SFT Evolution on TextCraft")
logger.info("=" * 60)
logger.info("Model: %s  Inference GPU: %s  Training GPU: %s", MODEL_PATH, INFERENCE_GPU, TRAINING_GPU)

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
            "Work backwards from the goal. Get raw materials first, then craft."
        ),
        default_max_tokens=256,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=200),
    planner=ReActPlanner(),
)

env = TextCraftEnv(max_steps_val=15, num_tasks=50)
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

    # Collect
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    good = [t for t in trajectories if t.success or t.total_reward > 0]
    logger.info("Collected: %d/%d positive", len(good), len(trajectories))

    if not good:
        logger.warning("No positive trajectories, skipping SFT")
        continue

    # Convert to SFT dataset
    sft_data = trajectories_to_sft_data(good, system_prompt=agent.brain.system_prompt)
    if not sft_data:
        continue
    dataset = to_hf_dataset(sft_data)
    logger.info("SFT dataset: %d samples", len(dataset))

    # Train LoRA
    hf = HFTrainingBackend(model_name=MODEL_PATH, device="cuda:1", torch_dtype="bfloat16")
    prev_adapter = OUTPUT_DIR / f"iter_{iteration-1}" / "adapter"
    model = hf.get_trainable_model(
        adapter_path=prev_adapter if prev_adapter.exists() else None,
        lora_config={"r": 16, "lora_alpha": 32,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    )
    tokenizer = hf.get_tokenizer()

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / f"iter_{iteration}" / "train"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_length=2048,
            save_strategy="no",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    logger.info("SFT loss: %.4f", train_result.training_loss if hasattr(train_result, 'training_loss') else 0)

    # Save and hot-swap
    adapter_path = OUTPUT_DIR / f"iter_{iteration}" / "adapter"
    hf.save_adapter(model, adapter_path)
    del model, trainer
    import torch; torch.cuda.empty_cache()

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
    "model": MODEL_PATH, "benchmark": "textcraft",
    "results": [{"stage": n, "success_rate": s, "avg_reward": r} for n, s, r in results],
}, indent=2))
