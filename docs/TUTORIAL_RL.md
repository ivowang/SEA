# Tutorial 4: RL Evolution (REINFORCE) on TextCraft

Train a local LLM via offline trajectory-level REINFORCE. Unlike SFT (which only learns from successes), REINFORCE learns from **both successes and failures** via advantage-weighted policy gradient.

**Requires GPU** — API for data collection, local GPU for REINFORCE training.

## Quick Start

```bash
python examples/rl_textcraft/run.py
```

## How It Works

### REINFORCE vs SFT
| | SFT | REINFORCE |
|---|---|---|
| **Data** | Only successful trajectories | Both successes AND failures |
| **Learning signal** | Imitation (copy demonstrations) | Policy gradient (maximize reward) |
| **Advantage** | N/A | G_t = discounted return per step |
| **Loss** | Cross-entropy on completions | -log_prob(action) × advantage |

### Three Phases
1. **Phase A: Collect trajectories via API** — 80 trajectories (need both successes and failures)
2. **Phase B: Baseline evaluation** — Load Qwen2.5-7B on vLLM (TP=2)
3. **Phase C: REINFORCE training loop** (3 iterations):
   - `trajectories_to_reinforce_data()` → context/action/advantage triples
   - Custom `compute_loss`: `-log_prob(action_tokens) × advantage`
   - Hot-swap adapter → evaluate → collect more data

### Why It Works
- Real environment rewards (success=1, failure=0) provide ground-truth signal
- Advantage normalization: successful trajectories get positive advantage, failures get negative
- Step-level credit assignment via discounted returns (gamma=0.99)
- Entropy bonus (0.01) encourages exploration during training

## Key Components

### RLEvolver (REINFORCE)
```python
from sea.evolution.methods.rl import RLEvolver

rl = RLEvolver(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    algorithm="reinforce",    # offline trajectory-level policy gradient
    device="cuda:7",
    gamma=0.99,               # discount factor
    entropy_coeff=0.01,       # exploration bonus
    learning_rate=1e-5,
    num_epochs=1,
    output_dir="outputs/tutorial_rl",
)
```

### The REINFORCE Loss
```python
# Per-sample: -sum(log_prob(action_tokens)) × advantage
# Where:
#   action_tokens = tokens after the prompt (labels != -100)
#   advantage = (G_t - mean(G)) / std(G)  (normalized return)
#   G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

## Expected Results

| Stage | Success Rate |
|-------|-------------|
| Baseline | ~20-30% |
| RL Iter 1 | ~30-40% |
| RL Iter 3 | ~40-55% |

Improvement: **+15-25%** from baseline.

## GPU Requirements

- **Inference**: GPU 4-5 (vLLM TP=2)
- **Training**: GPU 7 (REINFORCE, ~25GB)
