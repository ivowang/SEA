<p align="center">
  <img src="assets/logo.svg" alt="SEA Logo" width="280"/>
</p>

<h1 align="center">SEA: Self-Evolving Agent Platform</h1>

<p align="center">
  <b>A general-purpose research platform for building and studying Self-Evolving Agents.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#extending-sea">Extending SEA</a> &bull;
  <a href="docs/README_zh.md">中文文档</a>
</p>

---

## Why SEA?

Existing agent frameworks focus on *using* agents. SEA **evolves** them.

SEA decouples **what evolves** (LoRA weights, prompts, memory, skills) from **how it evolves** (SFT, RL, in-context learning, prompt optimization), letting researchers freely combine evolution methods and targets in a single config file.

```
┌─────────────────────────────────────────────────────┐
│              EvolutionPipeline                      │
│        collect → evolve → evaluate → repeat         │
├──────────┬──────────┬──────────┬────────────────────┤
│ Evolvers │ Targets  │ Metrics  │ TrajectoryBuffer   │
│ SFT / RL │ LoRA /   │ Tracker +│ Collect / Filter / │
│ ICL /... │ Prompt / │ Evaluator│ Sample             │
│          │ Memory / │          │                    │
│          │ Skill    │          │                    │
├──────────┴──────────┴──────────┴────────────────────┤
│                    SEAAgent                         │
│  ┌──────┐ ┌───────┐ ┌───────┐ ┌──────┐ ┌────────┐   │
│  │LLM as│ │Memory │ │Planner│ │Skills│ │Tools   │   │
│  │Brain │ │Epi/Sem│ │ReAct/ │ │Lib   │ │Registry│   │
│  │      │ │/Work  │ │LATS   │ │      │ │        │   │
│  └──┬───┘ └───────┘ └───────┘ └──────┘ └────────┘   │
├─────┴───────────────────────────────────────────────┤
│  LLM Backend                                        │
│  GPU 0: vLLM inference + LoRA hot-swap              │
│  GPU 1: HF Trainer (PEFT + TRL) for SFT/RL          │
├─────────────────────────────────────────────────────┤
│  Environment Layer: SEAEnv (Gymnasium-style)        │
│  TextCraft │ ALFWorld │ WebShop │ Custom …          │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
conda create -n sea python=3.11 -y
conda activate sea
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd SEA && pip install -r requirements.txt
```

### 2. Minimal Example

```python
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.env.wrapper import FunctionEnv
from sea.core.types import Action, Observation
from sea.evolution.methods.icl import ICLEvolver
from sea.evolution.data.trajectory import TrajectoryCollector
from sea.metrics.tracker import MetricsTracker
from sea.metrics.evaluator import Evaluator
from sea.llm.api_backend import APIBackend

# 1. LLM backend — any OpenAI-compatible endpoint
backend = APIBackend(model="Qwen/Qwen2.5-7B-Instruct",
                     base_url="http://localhost:8000/v1")

# 2. Build agent
agent = SEAAgent(
    brain=LLMBrain(backend=backend),
    memory=EpisodicMemory(),
    planner=ReActPlanner(),
)

# 3. Define a simple environment
def reset_fn(**kw):
    return Observation(text="You are in a room. Find the key and open the door."), \
           {"task_id": "room_escape", "task_description": "Find key, open door."}

def step_fn(action):
    if "key" in action.text.lower() and "door" in action.text.lower():
        return Observation(text="You escaped!"), 1.0, True, False, {"success": True}
    if "key" in action.text.lower():
        return Observation(text="You found a key."), 0.1, False, False, {}
    return Observation(text="Nothing happens."), 0.0, False, False, {}

env = FunctionEnv(name="room_escape", reset_fn=reset_fn, step_fn=step_fn,
                  task_ids=["room_escape"], max_steps_val=10)

# 4. Collect → Evolve → Evaluate
collector = TrajectoryCollector()
trajectories = collector.collect(agent, [env], n=10)

evolver = ICLEvolver(max_reflections_per_step=3, max_exemplars=5)
metrics = MetricsTracker()
evolver.evolve(agent, agent.evolvable_components()["memory"], trajectories, metrics)

results = Evaluator(num_episodes_per_env=10).evaluate(agent, [env])
print(f"Success rate: {results.success_rate:.0%}")
```

### 3. Full evolution loop with YAML config

```bash
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

---

## Architecture

### Core Protocols

```python
class Evolvable(Checkpointable, Generic[T]):
    """A component that can be evolved. T = state type."""
    def get_evolvable_state(self) -> T: ...
    def set_evolvable_state(self, state: T) -> None: ...

class Evolver(Checkpointable):
    def evolve(self, agent, target: Evolvable, trajectories, metrics, **kwargs) -> None: ...
```

**Any evolver can work with any compatible target:**

| | LoRA (`Path`) | Prompt (`str`) | Memory (`list[dict]`) | Skills (`list[dict]`) |
|---|:---:|:---:|:---:|:---:|
| **SFT** | ✅ | — | — | — |
| **RL (GRPO/DPO)** | ✅ | — | — | — |
| **ICL (Reflexion)** | — | — | ✅ | ✅ |
| **Prompt Optimizer** | — | ✅ | — | — |

### Agent Components

```
SEAAgent
├── LLMBrain        — wraps LLMBackend, manages LoRA + system prompt (Evolvable)
├── Memory           — episodic / semantic (FAISS) / working (Evolvable)
├── Planner          — ReAct (default), extensible to LATS / ToT
├── SkillLibrary     — FAISS-indexed, text / code / composed skills (Evolvable)
└── ToolRegistry     — calculator, json_parser, custom tools
```

### LLM Backend — Inference/Training Split

| Component | GPU | Role |
|---|---|---|
| `VLLMBackend` | GPU 0 | Fast inference, LoRA hot-swap via `LoRARequest` |
| `HFTrainingBackend` | GPU 1 | PEFT + TRL for SFT / RL training (callbacks supported) |

### Benchmarks

| Benchmark | Install | Task Type | Status |
|---|---|---|---|
| **TextCraft** | `pip install textcraft` | Minecraft crafting | ✅ Verified |
| **ALFWorld** | `pip install alfworld` + `alfworld-download` | Household robot (6 types) | ✅ Verified |
| **WebShop** | `git clone` + `setup.sh` | E-commerce navigation | Adapter ready |

ALFWorld supports **task-type filtering** (pick, clean, heat, cool, examine, pick_two) for multi-task and continual learning experiments.

---

## Extending SEA

### Add a new Evolution Method

```python
@EVOLVER_REGISTRY.register("my_method")
class MyEvolver(Evolver):
    def evolve(self, agent, target, trajectories, metrics, **kwargs):
        state = target.get_evolvable_state()
        # ... your evolution logic ...
        target.set_evolvable_state(new_state)
```

### Add a new Environment

```python
@ENV_REGISTRY.register("my_env")
class MyEnv(SEAEnv):
    def reset(self, *, seed=None, task_id=None):
        return Observation(text="Start"), {"task_id": task_id, "task_type": "my_type"}
    def step(self, action):
        return Observation(text="Result"), reward, done, False, {"success": done}
```

### Add Composed Skills

```python
from sea.agent.skills.code_skill import ComposedSkill

skill = ComposedSkill(
    name="clean_and_place",
    description="Clean an object and place it somewhere",
    composition_plan="navigate(obj) → pick(obj) → clean(obj) → navigate(dest) → put(obj, dest)",
    sub_skills=["navigate", "pick_up", "clean", "put_on"],
)
agent.skill_library.add_skill(skill)
```

> See [Extending SEA in detail](docs/PROMPTS.md) for full prompt templates for Claude Code.

---

## Project Structure

```
sea/
├── core/             # Evolvable[T], Checkpointable, types, registry
├── agent/            # SEAAgent, Brain, Planner, Memory, Skills, Tools
├── llm/              # vLLM (LoRA hot-swap), API backend, HF training backend
├── evolution/
│   ├── targets/      # LoRA (multi-adapter), Prompt, Memory, Skill targets
│   ├── methods/      # SFT, RL (GRPO/DPO), ICL (Reflexion), Prompt, ExpeL
│   ├── data/         # Trajectory collection (task-type filter), reward, dataset
│   └── pipeline.py   # EvolutionPipeline: collect → evolve → evaluate → repeat
├── env/              # SEAEnv ABC + TextCraft, ALFWorld, WebShop adapters
├── metrics/          # MetricsTracker, Evaluator (fixed seed), reporters
└── utils/            # Config (OmegaConf), logging, serialization
```

---

## Documentation

| Document | Description |
|---|---|
| **[Prompt Templates](docs/PROMPTS.md)** | Vibe coding templates for Claude Code — implement ideas in minutes |
| **[SFT Tutorial](docs/TUTORIAL_SFT.md)** | LoRA SFT on TextCraft with local GPU |
| **[RL Tutorial](docs/TUTORIAL_RL.md)** | GRPO with environment-backed reward on TextCraft |
| **[Memory Tutorial](docs/TUTORIAL_MEMORY.md)** | ICL/Reflexion memory evolution on TextCraft |
| **[Skill Tutorial](docs/TUTORIAL_SKILLS.md)** | Custom skill evolver on TextCraft |
| **[E2E Demo Tutorial](docs/TUTORIAL.md)** | First evolution experiment walkthrough |
| **[中文文档](docs/README_zh.md)** | Full Chinese documentation |

---

## Scripts

| Script | Description |
|---|---|
| `python scripts/run_evolution.py --config <yaml>` | Run the full evolution loop |
| `python scripts/run_eval.py --config <yaml> --checkpoint <path>` | Evaluate a checkpoint |
| `python scripts/serve_model.py --model <name>` | Start a vLLM OpenAI-compatible server |
| `python scripts/collect_trajectories.py --config <yaml> -n 100` | Collect trajectories without training |

---

## Examples

| Example | Method | Target | Environment |
|---|---|---|---|
| `examples/sft_textcraft/` | SFT (LoRA) | LM weights | TextCraft |
| `examples/rl_textcraft/` | GRPO (LoRA) | LM weights | TextCraft |
| `examples/memory_textcraft/` | ICL (Reflexion) | Episodic Memory | TextCraft |
| `examples/skill_textcraft/` | Custom Evolver | Skill Library | TextCraft |
| `examples/expel_textcraft/` | ExpeL | Memory (rules) | TextCraft |

---

## Citation

```bibtex
@software{sea2025,
  title  = {SEA: Self-Evolving Agent Platform},
  author = {Ivo Wang},
  year   = {2025},
  url    = {https://github.com/ivowang/SEA},
}
```

## License

MIT
