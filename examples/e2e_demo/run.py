#!/usr/bin/env python3
"""End-to-end demo: evolve an agent with SFT on a simple task.

This script demonstrates the full SEA evolution loop:
1. Start vLLM inference on GPU 4
2. Build an agent with ReAct planner + episodic memory
3. Define a simple multi-step task environment
4. Collect trajectories
5. Train LoRA adapter via SFT on successful trajectories (GPU 5)
6. Hot-swap the new adapter into vLLM
7. Evaluate improvement

Usage:
    cd /root/SEA
    python examples/e2e_demo/run.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("e2e_demo")

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("SEA_MODEL_PATH", "/root/models/Qwen2.5-1.5B-Instruct")
INFERENCE_GPU = "4"   # vLLM inference
TRAINING_GPU = "5"    # SFT training
NUM_COLLECT = 16      # trajectories per iteration
NUM_ITERATIONS = 3    # evolution iterations
NUM_EVAL = 10         # eval episodes
OUTPUT_DIR = Path("outputs/e2e_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Define a simple multi-step environment
# ──────────────────────────────────────────────────────────────────────
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

TASKS = {
    "make_tea": {
        "description": "Make a cup of tea. Steps: boil water, get cup, add tea bag, pour water.",
        "required_steps": ["boil water", "get cup", "add tea bag", "pour water"],
    },
    "make_toast": {
        "description": "Make toast. Steps: get bread, put in toaster, wait, take out toast.",
        "required_steps": ["get bread", "put in toaster", "wait", "take out"],
    },
    "wash_hands": {
        "description": "Wash your hands. Steps: turn on faucet, apply soap, scrub hands, rinse, dry hands.",
        "required_steps": ["turn on", "soap", "scrub", "rinse", "dry"],
    },
    "send_email": {
        "description": "Send an email. Steps: open email app, write subject, write body, click send.",
        "required_steps": ["open email", "subject", "body", "send"],
    },
}

class SimpleTaskEnv(SEAEnv):
    """Multi-step procedural tasks. Agent must perform steps in rough order."""

    def __init__(self):
        self._task = None
        self._completed = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "simple_tasks"

    @property
    def max_steps(self) -> int:
        return 15

    def get_task_ids(self) -> list[str]:
        return list(TASKS.keys())

    def reset(self, *, seed=None, task_id=None):
        if seed is not None:
            random.seed(seed)
        task_id = task_id or random.choice(list(TASKS.keys()))
        self._task = TASKS[task_id]
        self._completed = []
        self._step_count = 0

        obs_text = (
            f"Task: {self._task['description']}\n"
            f"You have completed 0/{len(self._task['required_steps'])} steps.\n"
            f"What do you do?"
        )
        return Observation(text=obs_text), {
            "task_id": task_id,
            "task_description": self._task["description"],
        }

    def step(self, action):
        self._step_count += 1
        action_lower = action.text.lower()

        # Check which required steps are matched
        newly_completed = []
        for req in self._task["required_steps"]:
            if req not in self._completed and req in action_lower:
                newly_completed.append(req)

        self._completed.extend(newly_completed)
        all_done = len(self._completed) >= len(self._task["required_steps"])

        if newly_completed:
            reward = len(newly_completed) * 0.25
            obs_text = (
                f"Good! You completed: {', '.join(newly_completed)}.\n"
                f"Progress: {len(self._completed)}/{len(self._task['required_steps'])} steps done."
            )
        else:
            reward = 0.0
            obs_text = (
                f"That didn't match any required step.\n"
                f"Progress: {len(self._completed)}/{len(self._task['required_steps'])} steps done.\n"
                f"Hint: remaining steps involve keywords from the task description."
            )

        if all_done:
            obs_text += "\nTask completed successfully!"
            reward += 0.5

        terminated = all_done
        truncated = self._step_count >= self.max_steps
        info = {
            "step": self._step_count,
            "completed_steps": list(self._completed),
            "success": all_done,
        }
        return Observation(text=obs_text), reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# 2. Build components
# ──────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("SEA End-to-End Demo")
logger.info("=" * 60)

# --- vLLM backend ---
logger.info("Loading vLLM on GPU %s with model %s ...", INFERENCE_GPU, MODEL_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{INFERENCE_GPU},{TRAINING_GPU}"

from sea.llm.vllm_backend import VLLMBackend

backend = VLLMBackend(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    max_model_len=2048,
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
    device="cuda:0",  # maps to INFERENCE_GPU via CUDA_VISIBLE_DEVICES
    enforce_eager=True,  # disable CUDA graph to save memory on 40GB cards
)

# --- Agent ---
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.agent.tools.registry import ToolRegistry

tool_reg = ToolRegistry()

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a helpful agent that completes tasks step by step. "
            "Read the task description carefully. Each step involves a specific action keyword. "
            "Perform the steps in order. Include the exact keywords from the task."
        ),
        default_max_tokens=256,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=200),
    planner=ReActPlanner(),
    tool_registry=tool_reg,
)

# --- Environment ---
env = SimpleTaskEnv()

# --- Metrics ---
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator

metrics = MetricsTracker(reporters=[ConsoleReporter(print_every=1)])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)

# ──────────────────────────────────────────────────────────────────────
# 3. Baseline evaluation (before evolution)
# ──────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Phase 1: Baseline evaluation (no evolution)")
logger.info("=" * 60)

baseline = evaluator.evaluate(agent, [env])
metrics.log({
    "baseline/success_rate": baseline.success_rate,
    "baseline/avg_reward": baseline.avg_reward,
    "baseline/avg_steps": baseline.avg_steps,
}, step=0)
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ──────────────────────────────────────────────────────────────────────
# 4. Evolution loop: collect → SFT → evaluate
# ──────────────────────────────────────────────────────────────────────
from sea.evolution.data.trajectory import TrajectoryCollector, TrajectoryBuffer
from sea.evolution.data.dataset import trajectories_to_sft_data, to_hf_dataset
from sea.llm.hf_backend import HFTrainingBackend

collector = TrajectoryCollector(buffer=TrajectoryBuffer(max_size=500))

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d", iteration, NUM_ITERATIONS)
    logger.info("=" * 60)

    # --- Collect trajectories ---
    logger.info("Collecting %d trajectories ...", NUM_COLLECT)
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)

    success_count = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful (%.0f%%)",
                success_count, len(trajectories),
                100 * success_count / max(len(trajectories), 1))

    # Filter successful ones for SFT
    good_trajs = [t for t in trajectories if t.success]
    if not good_trajs:
        # Also include partially successful (reward > 0)
        good_trajs = [t for t in trajectories if t.total_reward > 0]
    if not good_trajs:
        logger.warning("No positive trajectories, skipping SFT this iteration")
        continue

    # --- Convert to SFT dataset ---
    sft_data = trajectories_to_sft_data(
        good_trajs,
        system_prompt=agent.brain.system_prompt,
    )
    if not sft_data:
        logger.warning("No SFT samples, skipping")
        continue

    dataset = to_hf_dataset(sft_data)
    logger.info("SFT dataset: %d samples from %d trajectories", len(dataset), len(good_trajs))

    # --- SFT Training ---
    logger.info("Starting SFT training on GPU 1 (mapped from GPU %s) ...", TRAINING_GPU)
    hf = HFTrainingBackend(
        model_name=MODEL_PATH,
        device="cuda:1",  # maps to TRAINING_GPU via CUDA_VISIBLE_DEVICES
        torch_dtype="bfloat16",
    )

    # Load model with LoRA
    adapter_path = OUTPUT_DIR / f"iter_{iteration}" / "adapter"
    prev_adapter = OUTPUT_DIR / f"iter_{iteration - 1}" / "adapter" if iteration > 1 else None
    if prev_adapter and prev_adapter.exists():
        model = hf.get_trainable_model(adapter_path=prev_adapter)
    else:
        model = hf.get_trainable_model(lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        })

    tokenizer = hf.get_tokenizer()

    from trl import SFTTrainer, SFTConfig

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / f"iter_{iteration}" / "train"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_length=1024,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else 0
    logger.info("SFT done. Loss: %.4f", train_loss)

    # Save adapter
    hf.save_adapter(model, adapter_path)

    # Free training GPU
    del model, trainer
    import torch
    torch.cuda.empty_cache()

    # --- Hot-swap LoRA into vLLM ---
    logger.info("Hot-swapping LoRA adapter ...")
    lora_name = f"iter_{iteration}"
    agent.brain.backend.load_lora(str(adapter_path), name=lora_name)
    agent.brain.lora_name = lora_name
    agent.brain.lora_path = str(adapter_path)
    logger.info("LoRA '%s' loaded from %s", lora_name, adapter_path)

    # --- Evaluate ---
    logger.info("Evaluating ...")
    eval_result = evaluator.evaluate(agent, [env])
    metrics.log({
        f"iter_{iteration}/success_rate": eval_result.success_rate,
        f"iter_{iteration}/avg_reward": eval_result.avg_reward,
        f"iter_{iteration}/avg_steps": eval_result.avg_steps,
        f"iter_{iteration}/train_loss": train_loss,
    }, step=iteration)

    logger.info(
        "Iter %d result: success=%.0f%%, reward=%.3f, steps=%.1f (baseline was %.0f%%)",
        iteration,
        eval_result.success_rate * 100,
        eval_result.avg_reward,
        eval_result.avg_steps,
        baseline.success_rate * 100,
    )

# ──────────────────────────────────────────────────────────────────────
# 5. Final summary
# ──────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Evolution complete!")
logger.info("=" * 60)
logger.info("Baseline:  success=%.0f%%  reward=%.3f",
            baseline.success_rate * 100, baseline.avg_reward)

final = evaluator.evaluate(agent, [env])
logger.info("Final:     success=%.0f%%  reward=%.3f",
            final.success_rate * 100, final.avg_reward)

improvement = final.success_rate - baseline.success_rate
logger.info("Change:    %+.0f%% success rate", improvement * 100)

# Save summary
summary = {
    "model": MODEL_PATH,
    "num_iterations": NUM_ITERATIONS,
    "baseline": {"success_rate": baseline.success_rate, "avg_reward": baseline.avg_reward},
    "final": {"success_rate": final.success_rate, "avg_reward": final.avg_reward},
    "improvement": improvement,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
logger.info("Summary saved to %s", OUTPUT_DIR / "summary.json")
