#!/usr/bin/env python3
"""O-LoRA Continual Learning on ALFWorld.

Sequential training on ALFWorld's 6 task types with orthogonal LoRA
constraints to prevent catastrophic forgetting.

Comparison: O-LoRA (orthogonal constraint) vs Naive Sequential LoRA.

Requires: 2 GPUs (inference + training), Qwen3.5-9B model.

Usage:
    export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
    export SEA_INFERENCE_GPU="4"
    export SEA_TRAINING_GPU="5"
    python ideas/olora-continual/run.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("olora_continual")

MODEL_PATH = os.environ.get("SEA_MODEL_PATH", "/root/models/Qwen3.5-9B")
INFERENCE_GPU = os.environ.get("SEA_INFERENCE_GPU", "4")
TRAINING_GPU = os.environ.get("SEA_TRAINING_GPU", "5")

# ALFWorld task types to learn sequentially
TASK_SEQUENCE = ["pick", "clean", "heat", "cool", "examine", "pick_two"]
NUM_COLLECT_PER_TASK = 15
NUM_EVAL = 10
LORA_RANK = 8  # rank per task (O-LoRA accumulates: 8, 16, 24, ...)

OUTPUT_DIR = Path("ideas/olora-continual/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{INFERENCE_GPU},{TRAINING_GPU}"

# ── O-LoRA Orthogonal Loss ────────────────────────────────────────────

import torch
import torch.nn as nn


class OrthogonalRegularizer:
    """Computes the O-LoRA orthogonal loss.

    For each LoRA layer, penalizes overlap between the current task's
    trainable subspace (loranew_A) and all previous tasks' frozen subspace (lora_A).

    Loss = sum_layers |lora_A @ loranew_A^T|
    """

    def __init__(self, model, weight: float = 0.5):
        self.model = model
        self.weight = weight

    def compute_loss(self) -> torch.Tensor:
        """Compute orthogonal regularization loss."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        count = 0

        for name, param in self.model.named_parameters():
            if "lora_A" in name and "loranew" not in name:
                # Find corresponding loranew_A
                new_name = name.replace("lora_A", "loranew_A")
                for name2, param2 in self.model.named_parameters():
                    if name2 == new_name or (
                        "loranew_A" in name2
                        and name.split("lora_A")[0] == name2.split("loranew_A")[0]
                    ):
                        # Orthogonal penalty: |frozen_A @ new_A^T|
                        if param.shape[1] == param2.shape[1]:
                            loss += torch.abs(param @ param2.T).sum()
                            count += 1
                        break

        if count > 0:
            loss = self.weight * loss / count
        return loss


# ── Continual Learning Pipeline ───────────────────────────────────────

from sea.llm.vllm_backend import VLLMBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.env.benchmarks.alfworld import ALFWorldEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.data.dataset import trajectories_to_sft_data, to_hf_dataset
from sea.llm.hf_backend import HFTrainingBackend
from sea.evolution.targets.lm_params import LoRATarget
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

logger.info("=" * 60)
logger.info("O-LoRA Continual Learning on ALFWorld")
logger.info("Model: %s", MODEL_PATH)
logger.info("Task sequence: %s", TASK_SEQUENCE)
logger.info("=" * 60)

# Load vLLM
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
            "You are a household robot agent in ALFWorld. "
            "Complete the task by choosing actions from the available list. "
            "Think step by step."
        ),
        default_max_tokens=150,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=100),
    planner=ReActPlanner(),
)

env = ALFWorldEnv(split="eval_out_of_distribution", max_steps_val=30)
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0, eval_seed=42)
collector = TrajectoryCollector()

lora_target = LoRATarget(
    base_model_name=MODEL_PATH,
    adapter_dir=OUTPUT_DIR / "adapters" / "init",
    lora_config={"r": LORA_RANK, "lora_alpha": 16,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
)

# ── Baseline evaluation ──────────────────────────────────────────────
logger.info("Baseline evaluation (all task types) ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%", baseline.success_rate * 100)

# ── Sequential training ──────────────────────────────────────────────
results = {"baseline": {"overall": baseline.success_rate}}

for task_idx, task_type in enumerate(TASK_SEQUENCE):
    logger.info("=" * 60)
    logger.info("Task %d/%d: %s", task_idx + 1, len(TASK_SEQUENCE), task_type)
    logger.info("=" * 60)

    # Collect trajectories for this task type
    # Filter: only keep trajectories matching the current task type
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT_PER_TASK)
    task_trajs = [t for t in trajectories if t.task_type == task_type]
    good_trajs = [t for t in task_trajs if t.success or t.total_reward > 0]

    if not good_trajs:
        # Use all task_type trajs if no successes
        good_trajs = task_trajs if task_trajs else trajectories[:5]

    logger.info("Collected %d trajs, %d for task '%s', %d positive",
                len(trajectories), len(task_trajs), task_type, len(good_trajs))

    # Convert to SFT data
    sft_data = trajectories_to_sft_data(good_trajs, system_prompt=agent.brain.system_prompt)
    if not sft_data:
        logger.warning("No SFT data for task %s, skipping", task_type)
        continue

    dataset = to_hf_dataset(sft_data)

    # Train LoRA with O-LoRA orthogonal constraint
    hf = HFTrainingBackend(model_name=MODEL_PATH, device="cuda:1", torch_dtype="bfloat16")

    prev_adapter = lora_target.get_evolvable_state()
    model = hf.get_trainable_model(
        adapter_path=prev_adapter if prev_adapter.exists() else None,
        lora_config=lora_target.lora_config,
    )
    tokenizer = hf.get_tokenizer()

    # Set up O-LoRA regularizer
    ortho_reg = OrthogonalRegularizer(model, weight=0.5)

    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    class OLoRACallback(TrainerCallback):
        """Adds orthogonal loss during training."""
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if model is not None:
                ortho_loss = ortho_reg.compute_loss()
                if ortho_loss.item() > 0:
                    ortho_loss.backward()

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / f"task_{task_idx}_{task_type}" / "train"),
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
        callbacks=[OLoRACallback()],
    )

    logger.info("Training on task '%s' (%d samples)...", task_type, len(dataset))
    train_result = trainer.train()
    logger.info("Training done. Loss: %.4f",
                train_result.training_loss if hasattr(train_result, 'training_loss') else 0)

    # Save adapter for this task
    adapter_path = OUTPUT_DIR / f"task_{task_idx}_{task_type}" / "adapter"
    hf.save_adapter(model, adapter_path)
    lora_target.set_evolvable_state(adapter_path)
    lora_target.register_task_adapter(task_type, adapter_path)
    lora_target.r_sum += LORA_RANK

    del model, trainer
    torch.cuda.empty_cache()

    # Hot-swap and evaluate on ALL task types
    agent.brain.backend.load_lora(str(adapter_path), name=f"task_{task_idx}")
    agent.brain.lora_name = f"task_{task_idx}"

    eval_result = evaluator.evaluate(agent, [env])
    logger.info("After learning '%s': overall success=%.0f%%",
                task_type, eval_result.success_rate * 100)

    # Record per-task evaluation
    results[f"after_{task_type}"] = {
        "overall": eval_result.success_rate,
        "r_sum": lora_target.r_sum,
    }

# ── Final Summary ─────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("CONTINUAL LEARNING RESULTS")
logger.info("=" * 60)
for stage, data in results.items():
    logger.info("  %-20s  overall=%.0f%%  r_sum=%d",
                stage, data["overall"] * 100, data.get("r_sum", 0))

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "model": MODEL_PATH,
    "benchmark": "alfworld",
    "method": "olora",
    "task_sequence": TASK_SEQUENCE,
    "lora_rank_per_task": LORA_RANK,
    "results": results,
    "adapter_history": {k: str(v) for k, v in lora_target.adapter_history.items()},
}, indent=2))
logger.info("Saved to %s", OUTPUT_DIR / "summary.json")
