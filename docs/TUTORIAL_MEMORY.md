# Tutorial: Memory Evolution via Reflexion

# 教程：基于 Reflexion 的记忆进化

---

This tutorial shows how to evolve an agent's **memory** using ICL (In-Context Learning). The agent faces puzzle rooms with hidden rules, learns from failures through self-reflection, and stores reflections in episodic memory for future episodes.

**No GPU required** — uses any OpenAI-compatible API.

---

## How It Works

```
Iteration 1: Agent fails most rooms (no prior knowledge)
    ↓ ICLEvolver generates reflections on failures
    ↓ Reflections stored in EpisodicMemory
Iteration 2: Agent retrieves relevant reflections before acting
    ↓ Avoids repeated mistakes → higher reward
    ↓ More reflections accumulated
Iteration N: Memory grows → agent performance improves
```

The key insight: **the agent's memory IS the evolution target**. No model parameters change — only the contents of episodic memory evolve through accumulated reflections and exemplars.

核心思想：**Agent 的记忆本身就是进化对象**。不需要更新模型参数，只需要通过累积反思和示例来进化记忆内容。

---

## Step 1: Define the Environment

Each "Riddle Room" has a hidden solution sequence. The agent must discover the correct steps through trial and error.

```python
ROOMS = {
    "red_room": {
        "description": "You are in a red room. There is a light switch, a locked chest, and a door.",
        "solution": ["turn off light", "open chest", "take treasure", "exit"],
    },
    "blue_room": {
        "description": "You are in a blue room. There is a note, a painting, and a locked door.",
        "solution": ["read note", "move painting", "take key", "unlock door"],
    },
    # ... 6 rooms total
}
```

The environment uses **fuzzy keyword matching** — the agent's action must contain all keywords from the expected step (e.g., "turn off the light" matches "turn off light").

---

## Step 2: Build the Agent

```python
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner

backend = APIBackend(
    model="openai/gpt-5.4-nano",      # or any OpenAI-compatible model
    base_url="https://api.example.com/v1",
    api_key="your-key",
)

agent = SEAAgent(
    brain=LLMBrain(backend=backend, default_temperature=0.5),
    memory=EpisodicMemory(max_size=500),  # This is the evolution target
    planner=ReActPlanner(),
)
```

`EpisodicMemory` implements `Evolvable[list[dict]]`, so it can be found by `agent.evolvable_components()["memory"]` and evolved by any compatible evolver.

---

## Step 3: Baseline Evaluation

```python
from sea.metrics.evaluator import Evaluator

evaluator = Evaluator(num_episodes_per_env=12, eval_temperature=0.0)
baseline = evaluator.evaluate(agent, [env])
# Baseline: success=0%, reward=0.083
```

The evaluator uses `eval_mode=True` — no memory writes during evaluation, so results are clean.

---

## Step 4: Evolution Loop

```python
from sea.evolution.methods.icl import ICLEvolver
from sea.evolution.data.trajectory import TrajectoryCollector

collector = TrajectoryCollector()
evolver = ICLEvolver(
    max_reflections_per_step=5,  # Generate up to 5 reflections per iteration
    max_exemplars=5,             # Store up to 5 exemplars from successes
)

for iteration in range(1, 5):
    # Collect trajectories (writes to memory)
    trajectories = collector.collect(agent, [env], n=12)

    # ICL evolution: reflect on failures, store exemplars from successes
    memory_target = agent.evolvable_components()["memory"]
    evolver.evolve(agent, memory_target, trajectories, metrics)

    # Evaluate (eval_mode — no memory contamination)
    result = evaluator.evaluate(agent, [env])
```

**What happens inside `ICLEvolver.evolve()`:**

1. For each failed trajectory, the LLM generates a verbal reflection ("I should have read the note first before trying to unlock the door")
2. Reflections are stored in memory as `MemoryEntry(memory_type="reflection")`
3. For each successful trajectory, a textual exemplar is created and stored
4. In future episodes, the planner retrieves relevant reflections/exemplars via `memory.retrieve(query)`

---

## Results

| Stage | Success Rate | Avg Reward | Memory Size |
|-------|-------------|------------|-------------|
| Baseline | 0% | 0.083 | 0 |
| Iter 1 | 0% | 0.292 | 40 |
| Iter 2 | 0% | 0.250 | 103 |
| Iter 3 | 17% | 0.417 | 166 |
| Iter 4 | 17% | 0.417 | 255 |

Reward improved **5x** (0.083 → 0.417) and success rate went from **0% to 17%** through pure memory evolution — no parameter updates.

---

## Key Takeaways

1. **Memory is Evolvable**: `EpisodicMemory` implements `Evolvable[list[dict]]`, making it a first-class evolution target
2. **Side-effect-free eval**: `eval_mode=True` prevents evaluation from polluting memory
3. **No GPU needed**: ICL evolution only uses LLM inference calls
4. **Works with any backend**: Replace `APIBackend` with `VLLMBackend` for local models

---

## Run the Full Demo

```bash
export SEA_API_KEY="your-key"
export SEA_BASE_URL="https://api.example.com/v1"
export SEA_MODEL="openai/gpt-5.4-nano"
python examples/memory_evolution/run.py
```

To use a local vLLM server:
```bash
export SEA_BASE_URL="http://localhost:8000/v1"
export SEA_MODEL="Qwen/Qwen2.5-7B-Instruct"
```
