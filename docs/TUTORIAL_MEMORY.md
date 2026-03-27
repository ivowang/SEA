# Tutorial: Memory Evolution on TextCraft

Evolve an agent's memory via ICL (Reflexion) on the TextCraft benchmark. The agent attempts crafting tasks, reflects on failures ("I needed planks before crafting the sign"), and stores reflections in episodic memory. Accumulated reflections help in future episodes.

**No GPU required** — uses any OpenAI-compatible API.

## Quick Run

```bash
export SEA_API_KEY="your-key"
export SEA_BASE_URL="https://api.example.com/v1"
export SEA_MODEL="openai/gpt-5.4-nano"
python examples/memory_textcraft/run.py
```

To use local vLLM instead:
```bash
export SEA_BASE_URL="http://localhost:8000/v1"
export SEA_MODEL="Qwen/Qwen3.5-9B"
```

## How It Works

```
Iteration 1: Agent tries crafting tasks (no prior knowledge)
    → Some succeed, most fail on complex recipes
    ↓ ICLEvolver generates reflections on failures
    ↓ Stores exemplars from successes in EpisodicMemory
Iteration N: Agent retrieves relevant reflections before acting
    → "Last time I failed because I didn't get oak logs first"
    → Avoids repeated mistakes
```

## Results on TextCraft (gpt-5.4-nano)

| Stage | Success Rate | Memory Size |
|-------|-------------|-------------|
| Baseline | 80% | 0 |
| Iter 1 | 70% | ~120 |
| Iter 2 | 40% | ~250 |
| Iter 3 | 70% | ~380 |
| Iter 4 | 70% | 500 |

TextCraft with gpt-5.4-nano already has high baseline success (80%) on simpler recipes, leaving limited room for memory-based improvement. With a weaker model or harder tasks, the memory evolution effect is more pronounced.

## Key Code

```python
from sea.evolution.methods.icl import ICLEvolver
from sea.agent.memory.episodic import EpisodicMemory

agent = SEAAgent(
    brain=LLMBrain(backend=backend),
    memory=EpisodicMemory(max_size=500),  # evolution target
    planner=ReActPlanner(),
)

evolver = ICLEvolver(max_reflections_per_step=5, max_exemplars=3)

# Evolution loop
for iteration in range(4):
    trajectories = collector.collect(agent, [env], n=12)
    memory_target = agent.evolvable_components()["memory"]
    evolver.evolve(agent, memory_target, trajectories, metrics)
    result = evaluator.evaluate(agent, [env])  # eval_mode — no memory writes
```

## What ICLEvolver Does

1. **Failed trajectories** → LLM generates verbal reflections:
   > "I tried to craft oak_planks but I didn't have oak_log. Next time, get oak_log first."
2. **Successful trajectories** → stored as exemplars:
   > "Example for seed_42: get 1 oak log → craft 4 oak planks → craft 1 oak sign"
3. **Memory retrieval** → ReActPlanner retrieves relevant reflections before each action

## Full Script

See `examples/memory_textcraft/run.py`
