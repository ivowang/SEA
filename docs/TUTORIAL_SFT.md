# Tutorial 3: SFT Evolution (LoRA Fine-Tuning) on TextCraft

Train a local LLM (Qwen2.5-7B) via supervised fine-tuning on successful TextCraft trajectories. Data is collected via API, then used to train a LoRA adapter that is hot-swapped into the vLLM inference engine.

**Requires GPU** — API for data collection, local GPU for LoRA training.

## Quick Start

```bash
python examples/sft_textcraft/run.py
```

## How It Works

### Three Phases
1. **Phase A: Collect training data via API** — 50 trajectories with `gpt-5.4-nano`, filter successful ones
2. **Phase B: Baseline evaluation** — Load Qwen2.5-7B on vLLM (TP=2), evaluate 20 episodes
3. **Phase C: SFT training loop** (3 iterations) — Train LoRA → hot-swap → evaluate → collect more data

### Architecture
```
Phase A: APIBackend → gpt-5.4-nano → TextCraft → successful trajectories
Phase B: vLLM (GPU 4-5) → Qwen2.5-7B → baseline eval
Phase C: HFTrainingBackend (GPU 6) → LoRA training → hot-swap to vLLM → eval
```

### Why It Works
- API-generated successful trajectories provide **high-quality demonstrations**
- SFT teaches the local model the correct ReAct format + TextCraft action patterns
- `completion_only_loss=True` ensures the model only learns assistant responses
- LoRA (r=16) is lightweight — fits on a single A800 40GB

## Key Components

### SFTEvolver
```python
from sea.evolution.methods.sft import SFTEvolver

sft = SFTEvolver(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda:6",           # training GPU
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=4,
    max_length=1024,
    output_dir="outputs/tutorial_sft",
)
```

### LoRA Target
```python
from sea.evolution.targets.lm_params import LoRATarget

lora_target = LoRATarget(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    adapter_dir="outputs/tutorial_sft/adapter_init",
)
```

### Hot-Swap Flow
```python
# SFTEvolver.evolve() internally:
# 1. Load base model + LoRA on training GPU
# 2. Train on successful trajectories
# 3. Save adapter checkpoint
# 4. agent.brain.swap_lora(adapter_path) → update vLLM inference
```

## Expected Results

| Stage | Success Rate |
|-------|-------------|
| Baseline (no LoRA) | ~20-30% |
| SFT Iter 1 | ~40-50% |
| SFT Iter 3 | ~50-65% |

Improvement: **+20-35%** from baseline.

## GPU Requirements

- **Inference**: GPU 4-5 (vLLM TP=2, ~20GB per GPU)
- **Training**: GPU 6 (PEFT LoRA, ~25GB)
