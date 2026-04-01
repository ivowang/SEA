#!/usr/bin/env python3
"""Tutorial 3: SFT Evolution (LoRA Fine-Tuning) on TextCraft.

Demonstrates supervised fine-tuning of a local LLM:
- Phase A: Collect training data via API with high concurrency (no GPU)
- Phase B: Single-GPU train/eval loop using HF model throughout
  (no vLLM — the same model is used for training and inference)

Only 1 GPU needed. The HF model stays loaded; LoRA adapters are
merged/swapped between train and eval stages.

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
GPU_DEVICE = "cuda:6"
NUM_COLLECT = 50
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


def build_hf_agent(adapter_path: str | None = None) -> SEAAgent:
    """Build agent using HF model for inference (same model used for training).

    This avoids loading vLLM separately — one model for both train and eval.
    """
    from sea.llm.hf_backend import HFTrainingBackend

    logger.info("Loading HF model %s on %s (adapter: %s)...",
                MODEL_NAME, GPU_DEVICE, adapter_path or "none")
    hf = HFTrainingBackend(
        model_name=MODEL_NAME,
        device=GPU_DEVICE,
        torch_dtype="bfloat16",
    )
    model = hf.get_trainable_model(adapter_path=adapter_path)
    tokenizer = hf.get_tokenizer()

    # Wrap HF model as a simple generate-only backend
    class _HFInferenceBackend:
        """Minimal LLMBackend wrapper around a HF model for inference."""
        def __init__(self, model, tokenizer, device):
            self._model = model
            self._tokenizer = tokenizer
            self._device = device
            self.model_name = MODEL_NAME

        def generate(self, messages, *, temperature=0.0, max_tokens=150,
                     stop=None, lora_name=None, **kwargs):
            from sea.core.types import GenerationOutput
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return GenerationOutput(text=text)

        def generate_batch(self, batches, **kw):
            return [self.generate(msgs, **kw) for msgs in batches]

        def supports_lora(self): return False
        def load_lora(self, *a, **kw): pass
        def unload_lora(self, *a, **kw): pass

    backend = _HFInferenceBackend(model, tokenizer, GPU_DEVICE)
    brain = LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                     default_max_tokens=150, default_temperature=0.0)
    return SEAAgent(brain=brain, memory=WorkingMemory(max_size=20), planner=ReActPlanner())


def main():
    logger.info("=" * 60)
    logger.info("Tutorial 3: SFT Evolution (LoRA Fine-Tuning)")
    logger.info("Single GPU: %s", GPU_DEVICE)
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

    # ── Phase B: Train/Eval loop ──
    for iteration in range(NUM_SFT_ITERS + 1):
        # --- Eval: load HF model (with adapter if available) ---
        adapter_path = None
        if iteration > 0:
            state = lora_target.get_evolvable_state()
            adapter_path = str(state) if state and Path(str(state)).exists() else None

        label = "Baseline" if iteration == 0 else f"SFT Iter {iteration}"
        logger.info("\n── %s: Evaluating ──", label)

        agent = build_hf_agent(adapter_path=adapter_path)
        sr = evaluator.evaluate(agent, [env]).success_rate
        logger.info("%s: success_rate=%.1f%%", label, sr * 100)
        results_table.append((label, sr))

        # Free eval model
        del agent
        gc.collect()
        torch.cuda.empty_cache()

        if iteration == NUM_SFT_ITERS:
            break

        # --- Train: SFTEvolver loads its own model internally ---
        logger.info("\n── SFT Training %d/%d ──", iteration + 1, NUM_SFT_ITERS)
        from unittest.mock import MagicMock
        dummy_agent = MagicMock()
        dummy_agent.brain = MagicMock()
        dummy_agent.brain.swap_lora = MagicMock()
        dummy_agent.brain.system_prompt = SYSTEM_PROMPT
        sft.evolve(dummy_agent, lora_target, successful, metrics)

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
