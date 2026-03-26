# Your First Self-Evolving Agent: A Step-by-Step Tutorial

# 第一个自进化 Agent 实验：手把手教程

---

This tutorial walks you through running your **first complete evolution experiment** on the SEA platform — from zero to a measurably improved agent. We will:

1. Set up the environment
2. Understand the core concepts
3. Define a task environment
4. Build an agent
5. Collect trajectories
6. Train a LoRA adapter via SFT
7. Hot-swap the adapter and evaluate improvement

By the end, you will have an agent whose task success rate improved from **67% to 100%** through self-evolution.

本教程将带你从零开始，在 SEA 平台上完成**第一个完整的 Agent 进化实验**。实验结束后，你将看到 Agent 的任务成功率从 **67% 提升到 100%**。

---

## Prerequisites / 前置要求

- **Hardware**: 2 idle GPUs (A100/A800, 40GB+ each). One for inference, one for training.
- **Software**: conda, CUDA 12.x
- **Model**: Qwen2.5-1.5B-Instruct (or any HuggingFace chat model)

> If you only have 1 GPU, you can time-share by running inference and training sequentially. If you have no GPU, skip to [Appendix: CPU-only ICL experiment](#appendix-cpu-only-icl-experiment).

---

## Step 1: Installation / 安装环境

```bash
# Create conda environment
conda create -n sea python=3.11 -y
conda activate sea

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install SEA and all dependencies
cd /root/SEA
pip install -e .

# Verify
python -c "import sea; import torch; print(f'SEA v{sea.__version__}, torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

Expected output:
```
SEA v0.1.0, torch 2.6.0+cu124, CUDA True
```

**Download a model** (if not already available):
```bash
# Option A: HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /root/models/Qwen2.5-1.5B-Instruct

# Option B: From ModelScope (for users in mainland China)
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir /root/models/Qwen2.5-1.5B-Instruct
```

---

## Step 2: Core Concepts / 核心概念

Before writing code, let's understand the 5 key abstractions:

| Concept | Class | Purpose |
|---|---|---|
| **Agent** | `SEAAgent` | The composed agent (brain + memory + planner + skills + tools) |
| **Environment** | `SEAEnv` | Where the agent acts. Returns observations and rewards. |
| **Evolvable Target** | `Evolvable[T]` | What gets evolved: LoRA weights, prompts, memory, skills |
| **Evolver** | `Evolver` | How things evolve: SFT, RL, ICL, prompt optimization |
| **Pipeline** | `EvolutionPipeline` | The loop: collect → evolve → evaluate → repeat |

The fundamental idea: **decouple what evolves from how it evolves**. An SFT evolver can train LoRA weights. An ICL evolver can evolve memory. A prompt evolver can optimize the system prompt. Mix and match freely.

核心思想：**将"进化什么"和"怎么进化"解耦**。SFT 可以训练 LoRA 权重，ICL 可以进化记忆，Prompt 优化器可以改进系统提示。自由组合。

---

## Step 3: Define Your Environment / 定义环境

Every experiment starts with an environment. SEA environments follow the Gymnasium pattern:

```python
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

class SimpleTaskEnv(SEAEnv):
    """A multi-step procedural task environment."""

    TASKS = {
        "make_tea": {
            "description": "Make a cup of tea. Steps: boil water, get cup, add tea bag, pour water.",
            "required_steps": ["boil water", "get cup", "add tea bag", "pour water"],
        },
        "make_toast": {
            "description": "Make toast. Steps: get bread, put in toaster, wait, take out toast.",
            "required_steps": ["get bread", "put in toaster", "wait", "take out"],
        },
    }

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
        return list(self.TASKS.keys())

    def reset(self, *, seed=None, task_id=None):
        import random
        task_id = task_id or random.choice(list(self.TASKS.keys()))
        self._task = self.TASKS[task_id]
        self._completed = []
        self._step_count = 0
        obs_text = (
            f"Task: {self._task['description']}\n"
            f"Progress: 0/{len(self._task['required_steps'])} steps.\n"
            f"What do you do?"
        )
        return Observation(text=obs_text), {
            "task_id": task_id,
            "task_description": self._task["description"],
        }

    def step(self, action):
        self._step_count += 1
        action_lower = action.text.lower()

        # Match action keywords against required steps
        newly_completed = [
            req for req in self._task["required_steps"]
            if req not in self._completed and req in action_lower
        ]
        self._completed.extend(newly_completed)
        all_done = len(self._completed) >= len(self._task["required_steps"])

        reward = len(newly_completed) * 0.25 + (0.5 if all_done else 0.0)
        obs = f"Progress: {len(self._completed)}/{len(self._task['required_steps'])} steps."
        if all_done:
            obs += " Task completed!"

        return (
            Observation(text=obs),
            reward,
            all_done,                                  # terminated
            self._step_count >= self.max_steps,        # truncated
            {"success": all_done},
        )
```

**Key points / 要点:**
- `reset()` returns `(Observation, info_dict)` — info must contain `task_description`
- `step()` returns `(Observation, reward, terminated, truncated, info)` — same as Gymnasium
- `get_task_ids()` returns available tasks for the collector to sample from

---

## Step 4: Build the Agent / 构建 Agent

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # GPU 4 = inference, GPU 5 = training

from sea.llm.vllm_backend import VLLMBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner

# 1. LLM Backend — vLLM on GPU 0 (mapped from physical GPU 4)
backend = VLLMBackend(
    model="/root/models/Qwen2.5-1.5B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    max_model_len=2048,
    enable_lora=True,
    dtype="bfloat16",
    device="cuda:0",
    enforce_eager=True,  # saves memory on 40GB cards
)

# 2. Assemble the agent
agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a helpful agent that completes tasks step by step. "
            "Include the exact action keywords from the task description."
        ),
        default_max_tokens=256,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=200),
    planner=ReActPlanner(),
)

env = SimpleTaskEnv()
```

**What just happened / 发生了什么:**
- vLLM loads Qwen2.5-1.5B onto GPU 0 with LoRA support enabled
- `LLMBrain` wraps the backend and is itself `Evolvable` — its LoRA adapter and system prompt can both be evolved
- `EpisodicMemory` is also `Evolvable` — ICL evolvers can add reflections
- `ReActPlanner` uses the Thought → Action → Observation loop

---

## Step 5: Baseline Evaluation / 基线评测

Before evolving, measure the starting performance:

```python
from sea.metrics.evaluator import Evaluator

evaluator = Evaluator(num_episodes_per_env=10, eval_temperature=0.0)
baseline = evaluator.evaluate(agent, [env])

print(f"Baseline: success={baseline.success_rate:.0%}, reward={baseline.avg_reward:.3f}")
# Example output: Baseline: success=67%, reward=0.333
```

---

## Step 6: The Evolution Loop / 进化循环

This is the core of SEA. Each iteration: **collect → train → swap → evaluate**.

```python
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.evolution.data.dataset import trajectories_to_sft_data, to_hf_dataset
from sea.llm.hf_backend import HFTrainingBackend
from sea.metrics.tracker import MetricsTracker
from trl import SFTTrainer, SFTConfig
from pathlib import Path
import torch

collector = TrajectoryCollector()
metrics = MetricsTracker()
output_dir = Path("outputs/tutorial")
output_dir.mkdir(parents=True, exist_ok=True)

NUM_ITERATIONS = 3

for iteration in range(1, NUM_ITERATIONS + 1):
    print(f"\n{'='*50}")
    print(f"  Iteration {iteration}")
    print(f"{'='*50}")

    # ── Step A: Collect trajectories ──────────────────
    trajectories = collector.collect(agent, [env], n=16)
    good = [t for t in trajectories if t.success or t.total_reward > 0]
    print(f"Collected: {len(good)}/{len(trajectories)} positive trajectories")

    if not good:
        print("No positive trajectories. Skipping.")
        continue

    # ── Step B: Convert to SFT dataset ────────────────
    sft_data = trajectories_to_sft_data(good, system_prompt=agent.brain.system_prompt)
    dataset = to_hf_dataset(sft_data)

    # ── Step C: SFT training (GPU 1) ─────────────────
    hf = HFTrainingBackend(
        model_name="/root/models/Qwen2.5-1.5B-Instruct",
        device="cuda:1",
        torch_dtype="bfloat16",
    )

    prev_adapter = output_dir / f"iter_{iteration-1}" / "adapter"
    model = hf.get_trainable_model(
        adapter_path=prev_adapter if prev_adapter.exists() else None,
        lora_config={"r": 16, "lora_alpha": 32,
                     "target_modules": ["q_proj","k_proj","v_proj","o_proj"]},
    )
    tokenizer = hf.get_tokenizer()

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir / f"iter_{iteration}" / "train"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_length=1024,
            save_strategy="no",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    print(f"SFT loss: {result.training_loss:.4f}")

    # Save adapter
    adapter_path = output_dir / f"iter_{iteration}" / "adapter"
    hf.save_adapter(model, adapter_path)
    del model, trainer
    torch.cuda.empty_cache()

    # ── Step D: Hot-swap LoRA into vLLM ──────────────
    agent.brain.backend.load_lora(str(adapter_path), name=f"iter_{iteration}")
    agent.brain.lora_name = f"iter_{iteration}"
    print(f"LoRA hot-swapped: {adapter_path}")

    # ── Step E: Evaluate ─────────────────────────────
    result = evaluator.evaluate(agent, [env])
    print(f"After iter {iteration}: success={result.success_rate:.0%}, "
          f"reward={result.avg_reward:.3f} (baseline was {baseline.success_rate:.0%})")
```

**Expected output / 预期输出:**

```
==================================================
  Iteration 1
==================================================
Collected: 10/16 positive trajectories
SFT loss: 1.0356
LoRA hot-swapped: outputs/tutorial/iter_1/adapter
After iter 1: success=75%, reward=1.000 (baseline was 67%)

==================================================
  Iteration 2
==================================================
Collected: 15/16 positive trajectories
SFT loss: 1.1878
LoRA hot-swapped: outputs/tutorial/iter_2/adapter
After iter 2: success=100%, reward=1.062 (baseline was 67%)

==================================================
  Iteration 3
==================================================
Collected: 16/16 positive trajectories
SFT loss: 1.2276
LoRA hot-swapped: outputs/tutorial/iter_3/adapter
After iter 3: success=100%, reward=0.875 (baseline was 67%)
```

**What's happening at each step / 每步发生了什么:**

| Step | What | Where | Key API |
|---|---|---|---|
| **A. Collect** | Agent interacts with env, produces trajectories | GPU 0 (vLLM) | `collector.collect(agent, [env], n=16)` |
| **B. Convert** | Successful trajectories → chat-format dataset | CPU | `trajectories_to_sft_data()` → `to_hf_dataset()` |
| **C. Train** | LoRA SFT on the dataset | GPU 1 (HF Trainer) | `SFTTrainer.train()` |
| **D. Swap** | New LoRA checkpoint → hot-swap into vLLM | GPU 0 | `backend.load_lora(path)` |
| **E. Evaluate** | Run greedy eval episodes, measure improvement | GPU 0 | `evaluator.evaluate(agent, [env])` |

---

## Step 7: Understand What Evolved / 理解进化了什么

After 3 iterations, the agent's LoRA adapter has been trained on its own successful experiences. Let's inspect what changed:

```python
# Check evolvable components
for name, component in agent.evolvable_components().items():
    print(f"{name}: {component.evolution_metadata()}")

# Check the adapter on disk
import os
adapter_dir = output_dir / "iter_3" / "adapter"
for f in os.listdir(adapter_dir):
    size = os.path.getsize(adapter_dir / f) / 1024
    print(f"  {f}: {size:.1f} KB")
```

The LoRA adapter is only ~17MB (vs 3GB for the full model), containing the **delta** learned from successful trajectories.

LoRA 适配器只有约 17MB（完整模型为 3GB），只包含从成功轨迹中学到的**增量**。

---

## What's Next / 下一步

### Try a different evolution method

Replace SFT with **ICL (Reflexion)** — no training needed:

```python
from sea.evolution.methods.icl import ICLEvolver

evolver = ICLEvolver(max_reflections_per_step=5, max_exemplars=10)
memory_target = agent.evolvable_components()["memory"]
evolver.evolve(agent, memory_target, trajectories, metrics)
```

### Try a different target

Evolve the **system prompt** instead of LoRA weights:

```python
from sea.evolution.methods.prompt_evolver import PromptEvolver
from sea.evolution.targets.prompt import PromptTarget

prompt_target = PromptTarget(prompt_text=agent.brain.system_prompt)
evolver = PromptEvolver(num_variants=3)
evolver.evolve(agent, prompt_target, trajectories, metrics)
agent.brain.system_prompt = prompt_target.prompt_text
```

### Use the YAML-driven pipeline

Instead of writing Python, use a config file:

```bash
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

### Plug in your own environment

Implement `SEAEnv` (see [README](README.md#environments)), register with `@ENV_REGISTRY.register("name")`, and reference in config.

---

## Appendix: CPU-only ICL Experiment

If you don't have a GPU, you can still run an ICL (Reflexion) evolution using any OpenAI-compatible API:

```python
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.evolution.methods.icl import ICLEvolver
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.evaluator import Evaluator
from sea.env.wrapper import FunctionEnv
from sea.core.types import Observation

# Use any OpenAI-compatible endpoint
backend = APIBackend(
    model="gpt-4o-mini",  # or a local Ollama model
    # base_url="http://localhost:11434/v1",  # for Ollama
)

agent = SEAAgent(
    brain=LLMBrain(backend=backend),
    memory=EpisodicMemory(),
    planner=ReActPlanner(),
)

# Quick env
def reset_fn(**kw):
    return Observation(text="Find the key and open the door."), {"task_id": "escape"}
def step_fn(action):
    if "key" in action.text.lower() and "door" in action.text.lower():
        return Observation(text="Escaped!"), 1.0, True, False, {"success": True}
    return Observation(text="Try again."), 0.0, False, False, {}

env = FunctionEnv("escape", reset_fn, step_fn, ["escape"], max_steps_val=5)

# Collect → Evolve (ICL) → Evaluate
collector = TrajectoryCollector()
trajs = collector.collect(agent, [env], n=10)

evolver = ICLEvolver(max_reflections_per_step=3, extract_skills=True)
metrics = MetricsTracker()
evolver.evolve(agent, agent.evolvable_components()["memory"], trajs, metrics)

result = Evaluator(num_episodes_per_env=10).evaluate(agent, [env])
print(f"Success rate: {result.success_rate:.0%}")
```

No GPU, no LoRA, no training loop — just reflection and memory curation.

无需 GPU，无需 LoRA，无需训练循环 —— 只有反思和记忆整理。

---

## Appendix: Full Running Script

The complete, runnable version of this tutorial is at:

```
examples/e2e_demo/run.py
```

Run it with:

```bash
conda activate sea
cd /root/SEA
python examples/e2e_demo/run.py
```

It takes ~15 minutes on 2x A800-40GB and produces results in `outputs/e2e_demo/`.
