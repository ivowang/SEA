#!/usr/bin/env python -u
"""Tutorial 4: RL Evolution (REINFORCE) on TextCraft.

Demonstrates offline trajectory-level REINFORCE:
- Phase A: Collect trajectories via API with high concurrency (no GPU)
- Phase B: Load model ONCE, then alternate train/eval on same model

Unlike SFT (which only learns from successes), REINFORCE learns from
BOTH successes and failures via advantage-weighted policy gradient.

Only 1 GPU needed. Model loaded once — LoRA weights update in-place.

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
import os
import sys
from pathlib import Path

# Uncomment below to use offline mode (requires complete local model cache)
# os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.working import WorkingMemory
from sea.agent.planner import ReActPlanner
from sea.core.types import Trajectory, GenerationOutput
from sea.env.benchmarks.textcraft import TextCraftEnv
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.methods.rl import RLEvolver
from sea.evolution.targets.lm_params import LoRATarget
from sea.llm.hf_backend import HFTrainingBackend
from sea.metrics.evaluator import Evaluator
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_DEVICE = "cuda:7"
NUM_COLLECT = 80
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


# ── Helpers ─────────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).parent / "rl_trajectories_cache.json"


def _save_trajectories(trajectories: list[Trajectory], path: Path) -> None:
    import json as _json
    data = []
    for t in trajectories:
        data.append({
            "task_id": t.task_id, "task_type": t.task_type,
            "total_reward": t.total_reward, "success": t.success,
            "metadata": t.metadata,
            "steps": [{"observation": s.observation.text,
                        "action": s.action.text, "action_type": s.action.action_type,
                        "action_metadata": {k: str(v)[:300] for k, v in s.action.metadata.items()},
                        "reward": s.reward, "done": s.done,
                        "info": {k: v for k, v in s.info.items() if isinstance(v, (str, int, float, bool))},
                        } for s in t.steps],
        })
    path.write_text(_json.dumps(data, ensure_ascii=False, indent=1))
    logger.info("Saved %d trajectories to %s", len(data), path)


def _load_trajectories(path: Path) -> list[Trajectory]:
    import json as _json
    from sea.core.types import Step, Observation, Action
    data = _json.loads(path.read_text())
    trajs = []
    for d in data:
        steps = [Step(observation=Observation(text=s["observation"]),
                       action=Action(text=s["action"], action_type=s.get("action_type", "text"),
                                     metadata=s.get("action_metadata", {})),
                       reward=s["reward"], done=s["done"], info=s.get("info", {}))
                  for s in d["steps"]]
        t = Trajectory(steps=steps, task_id=d["task_id"], task_type=d.get("task_type", ""),
                       total_reward=d["total_reward"], success=d["success"],
                       metadata=d.get("metadata", {}))
        trajs.append(t)
    logger.info("Loaded %d cached trajectories from %s", len(trajs), path)
    return trajs


def collect_api_data(n: int, max_workers: int = 30) -> list[Trajectory]:
    """Collect trajectories via API, with local cache to avoid re-collection."""
    if CACHE_FILE.exists():
        cached = _load_trajectories(CACHE_FILE)
        if len(cached) >= n:
            logger.info("Using cached data (%d trajectories, need %d)", len(cached), n)
            return cached[:n]
        logger.info("Cache has %d but need %d, collecting more...", len(cached), n)

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
        agent_factory=make_agent, env_factory=make_env,
        n=n, max_workers=max_workers,
    )
    n_success = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d success, %d fail (both needed for REINFORCE)",
                n_success, len(trajectories) - n_success)

    _save_trajectories(trajectories, CACHE_FILE)
    return trajectories


class HFInferenceBackend:
    """Wrap a HF model as an LLMBackend for agent inference."""

    def __init__(self, model, tokenizer, device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self.model_name = MODEL_NAME

    def generate(self, messages, *, temperature=0.0, max_tokens=150,
                 stop=None, lora_name=None, **kwargs):
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01), do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return GenerationOutput(text=text)

    def generate_batch(self, batches, **kw):
        return [self.generate(msgs, **kw) for msgs in batches]

    def supports_lora(self): return False
    def load_lora(self, *a, **kw): pass
    def unload_lora(self, *a, **kw): pass


def make_agent_from_model(model, tokenizer) -> SEAAgent:
    """Wrap existing HF model into an SEAAgent for evaluation."""
    backend = HFInferenceBackend(model, tokenizer, GPU_DEVICE)
    brain = LLMBrain(backend=backend, system_prompt=SYSTEM_PROMPT,
                     default_max_tokens=150, default_temperature=0.0)
    return SEAAgent(brain=brain, memory=WorkingMemory(max_size=20), planner=ReActPlanner())


# ── Main ────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Tutorial 4: RL Evolution (REINFORCE)")
    logger.info("Single GPU: %s", GPU_DEVICE)
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

    # ── Phase B: Load model ONCE, then train/eval loop ──
    logger.info("Loading model %s on %s...", MODEL_NAME, GPU_DEVICE)
    hf = HFTrainingBackend(model_name=MODEL_NAME, device=GPU_DEVICE, torch_dtype="bfloat16")
    model = hf.get_trainable_model(lora_config={"r": 16, "lora_alpha": 32,
                                                  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    tokenizer = hf.get_tokenizer()
    logger.info("Model loaded.")

    lora_target = LoRATarget(base_model_name=MODEL_NAME, adapter_dir=OUTPUT_DIR / "adapter_init")

    rl = RLEvolver(
        model_name=MODEL_NAME, algorithm="reinforce", device=GPU_DEVICE,
        learning_rate=1e-5, num_epochs=1, batch_size=4,
        gradient_accumulation_steps=4, max_seq_length=1024,
        gamma=0.99, entropy_coeff=0.01,
        output_dir=str(OUTPUT_DIR), torch_dtype="bfloat16",
    )

    results_table = []

    for iteration in range(NUM_RL_ITERS + 1):
        # --- Eval with current model ---
        label = "Baseline" if iteration == 0 else f"RL Iter {iteration}"
        logger.info("\n── %s: Evaluating ──", label)
        model.eval()
        agent = make_agent_from_model(model, tokenizer)
        sr = evaluator.evaluate(agent, [env]).success_rate
        logger.info("%s: success_rate=%.1f%%", label, sr * 100)
        results_table.append((label, sr))

        if iteration == NUM_RL_ITERS:
            break

        # --- Train on same model ---
        logger.info("\n── REINFORCE Training %d/%d ──", iteration + 1, NUM_RL_ITERS)
        model.train()
        rl.evolve(agent, lora_target, all_trajectories, metrics,
                  model=model, tokenizer=tokenizer)

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

    del model
    gc.collect()
    torch.cuda.empty_cache()
    env.close()
    logger.info("Done! LoRA adapter saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
