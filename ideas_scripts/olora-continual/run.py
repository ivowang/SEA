#!/usr/bin/env python3
"""O-LoRA Continual Learning on ALFWorld.

Phase 1: API concurrent collection of trajectories for all 6 task types
Phase 2: Local Qwen3.5-9B baseline evaluation (all task types)
Phase 3: Sequential O-LoRA SFT, with per-task-type evaluation after each stage

Produces an upper-triangular evaluation table showing forgetting.

Usage:
    export SEA_API_KEY="your-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"
    export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
    export CUDA_VISIBLE_DEVICES="6,7"
    python ideas/olora-continual/run.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("olora_continual")

# ── Config ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
API_MODEL = "openai/gpt-5.4-nano"
MODEL_PATH = os.environ.get("SEA_MODEL_PATH", "/root/models/Qwen3.5-9B")

TASK_SEQUENCE = ["pick", "clean", "heat", "cool", "examine", "pick_two"]
NUM_COLLECT_PER_TASK = 15
NUM_EVAL_PER_TYPE = 5  # eval episodes per task type
LORA_RANK = 8

OUTPUT_DIR = Path("ideas/olora-continual/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a household robot agent in ALFWorld. "
    "Complete the task by choosing actions from the available list. "
    "Think step by step."
)
LORA_CONFIG = {
    "r": LORA_RANK, "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# ── O-LoRA Orthogonal Loss ────────────────────────────────────────────
import torch
import torch.nn as nn


class OrthogonalRegularizer:
    """O-LoRA orthogonal loss: penalizes overlap between task subspaces."""

    def __init__(self, model, weight: float = 0.5):
        self.model = model
        self.weight = weight

    def compute_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        count = 0
        for name, param in self.model.named_parameters():
            if "lora_A" in name and "loranew" not in name:
                new_name = name.replace("lora_A", "loranew_A")
                for name2, param2 in self.model.named_parameters():
                    if name2 == new_name or (
                        "loranew_A" in name2
                        and name.split("lora_A")[0] == name2.split("loranew_A")[0]
                    ):
                        if param.shape[1] == param2.shape[1]:
                            loss += torch.abs(param @ param2.T).sum()
                            count += 1
                        break
        if count > 0:
            loss = self.weight * loss / count
        return loss


# ── Imports ───────────────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
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
from sea.core.types import Trajectory


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("O-LoRA Continual Learning on ALFWorld")
    logger.info("API model: %s (collection)", API_MODEL)
    logger.info("Local model: %s (training + eval)", MODEL_PATH)
    logger.info("Task sequence: %s", TASK_SEQUENCE)
    logger.info("=" * 60)

    env = ALFWorldEnv(split="eval_out_of_distribution", max_steps_val=30)
    metrics = MetricsTracker(reporters=[ConsoleReporter()])

    lora_target = LoRATarget(
        base_model_name=MODEL_PATH,
        adapter_dir=OUTPUT_DIR / "adapters" / "init",
        lora_config=LORA_CONFIG,
    )

    # ==================================================================
    # Phase 1: Collect trajectories for ALL task types via API
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Collecting trajectories via API (%s)", API_MODEL)
    logger.info("=" * 60)

    # Collect ALL episodes using subprocess parallelism (30 workers).
    # Each subprocess has its own ALFWorld + API client, fully isolated.
    total_episodes = NUM_COLLECT_PER_TASK * len(TASK_SEQUENCE)
    NUM_WORKERS = 30

    logger.info("Collecting %d episodes with %d parallel subprocess workers ...",
                total_episodes, NUM_WORKERS)

    all_trajs = TrajectoryCollector.collect_subprocess(
        n=total_episodes,
        num_workers=NUM_WORKERS,
        env_name="alfworld",
        env_kwargs={"split": "eval_out_of_distribution", "max_steps_val": 30},
        backend_kwargs={"model": API_MODEL, "base_url": BASE_URL, "api_key": API_KEY},
        system_prompt=SYSTEM_PROMPT,
        max_tokens=150,
        temperature=0.7,
    )

    # Group by task type
    all_trajectories: dict[str, list[Trajectory]] = {tt: [] for tt in TASK_SEQUENCE}
    for traj in all_trajs:
        tt = traj.task_type
        if tt in all_trajectories:
            all_trajectories[tt].append(traj)
        else:
            # Unknown type — assign to closest match or skip
            all_trajectories.setdefault("unknown", []).append(traj)

    traj_summary = {tt: {"count": len(ts), "success": sum(1 for t in ts if t.success)}
                    for tt, ts in all_trajectories.items() if ts}
    logger.info("Collection complete: %s", traj_summary)

    gc.collect()

    # ==================================================================
    # Phase 2: Baseline eval with local model (vLLM TP=2)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Baseline evaluation with local model")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "6,7")

    backend = VLLMBackend(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=64,
        dtype="bfloat16",
        device="cuda:0",
        enforce_eager=True,
    )

    local_agent = SEAAgent(
        brain=LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                       default_max_tokens=150, default_temperature=0.0),
        memory=EpisodicMemory(max_size=50),
        planner=ReActPlanner(),
    )

    evaluator = Evaluator(num_episodes_per_env=NUM_EVAL_PER_TYPE, eval_temperature=0.0, eval_seed=42)

    # Evaluate baseline on each task type
    eval_table: dict[str, dict[str, float]] = {}
    baseline_row: dict[str, float] = {}

    logger.info("Evaluating baseline on each task type ...")
    for tt in TASK_SEQUENCE:
        result = evaluator.evaluate(local_agent, [env])
        baseline_row[tt] = result.success_rate
        logger.info("  Baseline on '%s': %.0f%%", tt, result.success_rate * 100)

    eval_table["baseline"] = baseline_row
    logger.info("Baseline complete: %s",
                {k: f"{v:.0%}" for k, v in baseline_row.items()})

    # Shutdown vLLM for training
    logger.info("Shutting down vLLM for training phase ...")
    del backend, local_agent
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    # ==================================================================
    # Phase 3: Sequential O-LoRA SFT
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Phase 3: Sequential O-LoRA SFT")
    logger.info("=" * 60)

    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    current_lora_path = None

    for task_idx, task_type in enumerate(TASK_SEQUENCE):
        logger.info("=" * 60)
        logger.info("Training %d/%d: %s", task_idx + 1, len(TASK_SEQUENCE), task_type)
        logger.info("=" * 60)

        # Get data for this task
        task_trajs = all_trajectories.get(task_type, [])
        good_trajs = [t for t in task_trajs if t.success or t.total_reward > 0]
        if not good_trajs:
            good_trajs = task_trajs[:5] if task_trajs else []

        sft_data = trajectories_to_sft_data(good_trajs, system_prompt=SYSTEM_PROMPT)
        if not sft_data:
            logger.warning("No SFT data for '%s', skipping", task_type)
            eval_table[f"after_{task_type}"] = {tt: -1.0 for tt in TASK_SEQUENCE[:task_idx+1]}
            continue

        dataset = to_hf_dataset(sft_data)
        logger.info("SFT data: %d samples for '%s'", len(dataset), task_type)

        # ── Train on GPU 0 ────────────────────────────────────────────
        hf = HFTrainingBackend(model_name=MODEL_PATH, device="cuda:0", torch_dtype="bfloat16")
        prev_adapter = lora_target.get_evolvable_state()
        model = hf.get_trainable_model(
            adapter_path=prev_adapter if prev_adapter.exists() else None,
            lora_config=LORA_CONFIG,
        )
        tokenizer = hf.get_tokenizer()

        ortho_reg = OrthogonalRegularizer(model, weight=0.5)

        class OLoRACallback(TrainerCallback):
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

        logger.info("Training ...")
        train_result = trainer.train()
        loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0
        logger.info("Training done. Loss: %.4f", loss)

        adapter_path = OUTPUT_DIR / f"task_{task_idx}_{task_type}" / "adapter"
        hf.save_adapter(model, adapter_path)
        lora_target.set_evolvable_state(adapter_path)
        lora_target.register_task_adapter(task_type, adapter_path)
        lora_target.r_sum += LORA_RANK
        current_lora_path = str(adapter_path)

        del model, trainer, hf
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

        # ── Eval with vLLM TP=2 + new adapter ────────────────────────
        logger.info("Starting vLLM for evaluation ...")
        backend = VLLMBackend(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.80,
            max_model_len=4096,
            enable_lora=True,
            max_lora_rank=64,
            dtype="bfloat16",
            device="cuda:0",
            enforce_eager=True,
        )
        backend.load_lora(current_lora_path, name=f"task_{task_idx}")

        eval_agent = SEAAgent(
            brain=LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                           default_max_tokens=150, default_temperature=0.0,
                           lora_name=f"task_{task_idx}"),
            memory=EpisodicMemory(max_size=50),
            planner=ReActPlanner(),
        )

        # Eval on all SEEN task types (upper triangular)
        row: dict[str, float] = {}
        seen_tasks = TASK_SEQUENCE[:task_idx + 1]
        for tt in seen_tasks:
            result = evaluator.evaluate(eval_agent, [env])
            row[tt] = result.success_rate
            logger.info("  After '%s' training, eval on '%s': %.0f%%",
                        task_type, tt, result.success_rate * 100)

        eval_table[f"after_{task_type}"] = row

        # Shutdown vLLM
        del backend, eval_agent
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    # ==================================================================
    # Summary: Upper-triangular table
    # ==================================================================
    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING RESULTS (Upper-Triangular Table)")
    logger.info("=" * 60)

    header = f"{'Stage':<20}" + "".join(f"{tt:>10}" for tt in TASK_SEQUENCE)
    logger.info(header)
    logger.info("-" * len(header))

    for stage, row in eval_table.items():
        cells = []
        for tt in TASK_SEQUENCE:
            if tt in row and row[tt] >= 0:
                cells.append(f"{row[tt]*100:>9.0f}%")
            else:
                cells.append(f"{'—':>10}")
        logger.info(f"{stage:<20}" + "".join(cells))

    # Save results
    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "model": MODEL_PATH,
        "api_model": API_MODEL,
        "benchmark": "alfworld",
        "method": "olora",
        "task_sequence": TASK_SEQUENCE,
        "lora_rank_per_task": LORA_RANK,
        "eval_table": eval_table,
        "collection_summary": traj_summary,
    }, indent=2))
    logger.info("Saved to %s", OUTPUT_DIR / "summary.json")
