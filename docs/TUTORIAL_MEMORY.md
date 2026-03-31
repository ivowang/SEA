# Tutorial 1: Memory Evolution (ICL/Reflexion) on TextCraft

Evolve an agent's working memory via ICL (Reflexion). The agent reflects on failures ("I needed planks before crafting the sign"), stores reflections + successful exemplars in `WorkingMemory`, and uses them in future episodes.

**No GPU required** — uses OpenAI-compatible API (`gpt-5.4-nano`).

## Quick Start

```bash
python examples/memory_textcraft/run.py
```

## How It Works

### Architecture
```
SEAAgent
├── LLMBrain (APIBackend → gpt-5.4-nano)
├── WorkingMemory (max_size=50, Evolvable)
└── ReActPlanner
```

### Evolution Loop (5 iterations)
1. **Collect** 30 trajectories on TextCraft (mix of successes and failures)
2. **ICLEvolver** processes the batch:
   - Generates **reflections** on failed trajectories ("What went wrong?")
   - Curates **exemplars** from successful trajectories (step-by-step demonstrations)
   - Writes to memory via `target.set_evolvable_state()` (Evolvable contract)
3. **Evaluate** 20 episodes → measure success rate improvement

### Why It Works
- TextCraft tasks share common sub-recipes (planks, sticks, dyes)
- Reflections like "Always `get 1 oak log` before crafting planks" transfer across tasks
- Exemplars provide concrete successful action sequences as context
- `WorkingMemory.retrieve()` uses keyword matching to find relevant memories

## Key Components

### ICLEvolver
```python
from sea.evolution.methods.icl import ICLEvolver

evolver = ICLEvolver(
    max_reflections_per_step=5,   # reflections from failures
    max_exemplars=10,             # exemplars from successes
    exemplar_selection="diverse", # maximize coverage
)
```

### WorkingMemory (Evolvable)
```python
from sea.agent.memory.working import WorkingMemory

memory = WorkingMemory(max_size=50)
# Implements Evolvable[list[dict]] — ICLEvolver can read/write via:
#   target.get_evolvable_state() → list of memory entry dicts
#   target.set_evolvable_state(updated_entries)
```

### Evolution Target
```python
# Get the memory as an evolution target
memory_target = agent.evolvable_components()["memory"]
evolver.evolve(agent, memory_target, trajectories, metrics)
```

## Expected Results

| Stage | Success Rate | Memory Size |
|-------|-------------|-------------|
| Baseline | ~70-80% | 0 |
| Iter 1 | ~75-85% | 10-15 |
| Iter 3 | ~80-90% | 25-35 |
| Iter 5 | ~85-95% | 40-50 |

Improvement: **+10-20%** over 5 iterations.

## Customization

- **Memory size**: Increase `max_size` for more context, decrease to reduce noise
- **Reflection depth**: Adjust `max_reflections_per_step` (more = richer analysis)
- **Exemplar strategy**: `"diverse"` for broad coverage, `"highest_reward"` for quality
- **Different environment**: Replace `TextCraftEnv` with any `SEAEnv` implementation
