# SEA Platform Architecture

> A comprehensive guide to the Self-Evolving Agent platform for the research team.

## Overview

SEA is a modular research platform for studying how LLM-based agents can improve themselves through experience. The platform connects four major concerns:

1. **Agent** — An LLM-based agent that perceives, reasons, and acts in environments
2. **Environment** — Benchmarks where the agent performs tasks (TextCraft, ALFWorld, WebShop)
3. **Evolution** — Methods that improve the agent from collected trajectories (SFT, RL, ICL, ExpeL)
4. **Infrastructure** — LLM backends, metrics, and utilities that support the above

The central abstraction is the **evolution loop**: the agent acts in an environment, trajectories are collected, an evolver updates a target component (LoRA weights, memory, skills, prompts), and the improved agent acts again.

```
┌─────────────────────────────────────────────────────────────┐
│                      Evolution Loop                         │
│                                                             │
│   ┌─────────┐     ┌───────────┐     ┌──────────────┐       │
│   │  Agent   │────▶│Environment│────▶│  Trajectory   │      │
│   │(act/plan)│◀────│ (step)    │     │  Collection   │      │
│   └────┬─────┘     └───────────┘     └──────┬───────┘      │
│        │                                     │              │
│        │         ┌───────────┐               │              │
│        └─────────│  Evolver  │◀──────────────┘              │
│                  │(SFT/RL/ICL)│                              │
│                  └─────┬─────┘                              │
│                        │ updates                            │
│                  ┌─────▼─────┐                              │
│                  │  Target   │                              │
│                  │(LoRA/Mem/ │                              │
│                  │ Skills)   │                              │
│                  └───────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
sea/
├── core/                  # Foundation protocols and types
│   ├── base.py            # Checkpointable, Evolvable[T] ABCs
│   ├── types.py           # Observation, Action, Step, Trajectory, etc.
│   └── registry.py        # Plugin registry for envs, evolvers, backends
│
├── agent/                 # The agent and its components
│   ├── agent.py           # SEAAgent: orchestrates act/episode/checkpoint
│   ├── brain.py           # LLMBrain: LLM wrapper with LoRA management
│   ├── planner.py         # ReActPlanner: Thought/Action parsing
│   ├── memory/            # Memory subsystem
│   │   ├── base.py        # Memory ABC, MemoryEntry dataclass
│   │   ├── working.py     # WorkingMemory (default, Evolvable)
│   │   ├── episodic.py    # EpisodicMemory (optional extension)
│   │   └── semantic.py    # SemanticMemory (optional, FAISS-based)
│   ├── skills/            # Skill subsystem (SKILL.md paradigm)
│   │   ├── skill_md.py    # SKILL.md parser/renderer (YAML frontmatter)
│   │   ├── disclosure.py  # Progressive Disclosure (INDEX/SUMMARY/FULL)
│   │   ├── library.py     # SkillLibrary (file-system backed, Evolvable)
│   │   ├── base.py        # Legacy Skill ABC
│   │   └── code_skill.py  # Legacy TextSkill/CodeSkill/ComposedSkill
│   └── tools/             # Tool subsystem
│       ├── base.py        # Tool ABC, ToolResult
│       ├── registry.py    # ToolRegistry
│       ├── builtins.py    # CalculatorTool (AST-based safe eval)
│       └── skill_reader.py# ReadSkillTool (progressive disclosure)
│
├── env/                   # Environment adapters
│   ├── base.py            # SEAEnv ABC (reset/step/get_task_ids)
│   ├── benchmarks/
│   │   ├── textcraft.py   # TextCraft (Minecraft crafting)
│   │   ├── alfworld.py    # ALFWorld (household tasks, per-game task_id)
│   │   └── webshop.py     # WebShop (web shopping)
│   ├── wrapper.py         # GymnasiumWrapper, FunctionEnv
│   └── parallel.py        # ParallelEnvRunner (deprecated)
│
├── evolution/             # Evolution methods and pipeline
│   ├── base.py            # Evolver ABC
│   ├── pipeline.py        # EvolutionPipeline (main loop)
│   ├── methods/
│   │   ├── sft.py         # SFTEvolver (LoRA fine-tuning)
│   │   ├── rl.py          # RLEvolver (REINFORCE / DPO)
│   │   ├── icl.py         # ICLEvolver (Reflexion, exemplars)
│   │   ├── expel.py       # ExpeLEvolver (IF-THEN-BECAUSE rules)
│   │   └── prompt_evolver.py  # PromptEvolver (prompt optimization)
│   ├── data/
│   │   ├── trajectory.py  # TrajectoryCollector (serial/parallel/subprocess)
│   │   ├── dataset.py     # Trajectory→SFT/REINFORCE/DPO data conversion
│   │   ├── reward.py      # Reward functions (env, success, LLM judge)
│   │   └── parallel_worker.py  # Subprocess worker for collection
│   └── targets/
│       ├── lm_params.py   # LoRATarget (adapter path management)
│       ├── prompt.py       # PromptTarget (system prompt evolution)
│       ├── memory_target.py# MemoryTarget (wrapper)
│       └── skill_target.py # SkillTarget (wrapper)
│
├── llm/                   # LLM backends
│   ├── base.py            # LLMBackend ABC
│   ├── api_backend.py     # APIBackend (OpenAI-compatible, remote)
│   ├── vllm_backend.py    # VLLMBackend (local, LoRA hot-swap)
│   └── hf_backend.py      # HFTrainingBackend (PEFT LoRA training)
│
├── metrics/               # Evaluation and reporting
│   ├── evaluator.py       # Evaluator (run episodes, aggregate results)
│   ├── tracker.py         # MetricsTracker (collect and dispatch)
│   ├── builtin_metrics.py # Standard metric functions
│   └── reporters/
│       ├── console.py     # Console table output
│       ├── tensorboard.py # TensorBoard logging
│       └── wandb.py       # Weights & Biases logging
│
└── utils/                 # Shared utilities
    ├── config.py          # YAML config loading with OmegaConf
    ├── logging.py         # Logging setup
    └── serialization.py   # JSON serialization helpers
```

---

## Core Protocols

### Checkpointable

Every stateful component implements `Checkpointable`:

```python
class Checkpointable(ABC):
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
```

### Evolvable[T]

Components that can be improved by evolution methods implement `Evolvable[T]`, where `T` is the state type:

```python
class Evolvable(Checkpointable, Generic[T]):
    def get_evolvable_state(self) -> T: ...
    def set_evolvable_state(self, state: T) -> None: ...
    def evolution_metadata(self) -> dict[str, Any]: ...
```

| Component | T | Description |
|-----------|---|-------------|
| `LoRATarget` | `Path` | Path to LoRA adapter checkpoint |
| `PromptTarget` | `str` | System prompt text |
| `WorkingMemory` | `list[dict]` | Memory entries |
| `SkillLibrary` | `list[dict]` | Skill definitions |

### Registry

Components are discoverable via the registry pattern:

```python
@ENV_REGISTRY.register("textcraft")
class TextCraftEnv(SEAEnv): ...

# Later, in config:
env = ENV_REGISTRY.build("textcraft", max_steps_val=30)
```

Global registries: `ENV_REGISTRY`, `EVOLVER_REGISTRY`, `LLM_BACKEND_REGISTRY`, `MEMORY_REGISTRY`, `SKILL_REGISTRY`, `TOOL_REGISTRY`, `REWARD_REGISTRY`, `REPORTER_REGISTRY`.

---

## Agent Architecture

```
SEAAgent
├── LLMBrain          # LLM wrapper (generate, LoRA swap)
│   └── LLMBackend    # APIBackend / VLLMBackend / HFTrainingBackend
├── WorkingMemory      # Sliding window of recent memories (Evolvable)
├── ReActPlanner       # Thought→Action parsing, history management
├── SkillLibrary       # SKILL.md files with progressive disclosure
│   └── ReadSkillTool  # On-demand full skill content retrieval
└── ToolRegistry       # Calculator, ReadSkill, custom tools
```

### Action Loop (`agent.act()`)

```
1. Retrieve relevant memories (keyword matching)
2. Retrieve relevant skills (keyword/embedding, SUMMARY level)
3. Get skill index (INDEX level — names only)
4. Build PlanningContext with all of the above
5. Planner constructs prompt → LLM generates → parse Thought/Action
6. If tool_call → execute tool → feed result back → re-plan (max 3 rounds)
7. Return final Action to environment
```

### Episode Loop (`agent.run_episode()`)

```
env.reset(task_id) → observation
for each step:
    action = agent.act(observation)
    if action is "finish" → break
    observation, reward, done = env.step(action)
    record Step in Trajectory
return Trajectory(steps, reward, success)
```

---

## Skill System: SKILL.md Progressive Disclosure

Skills follow the modern SKILL.md paradigm with three-tier progressive disclosure:

### Format

```markdown
---
name: craft_oak_planks
description: Craft oak planks from oak logs
tags: [crafting, textcraft]
when_to_use: When you need oak planks
---

## Steps
1. Get oak logs: `get 1 oak logs`
2. Craft planks: `craft 4 oak planks using 1 oak logs`
```

### Disclosure Levels

| Level | Content | When Used |
|-------|---------|-----------|
| **INDEX** | name + description | Always in system prompt |
| **SUMMARY** | + when_to_use + steps outline | Top-k retrieved skills |
| **FULL** | Complete .md body | On-demand via `read_skill` tool |

### Storage

Skills are stored as `.md` files on disk (version-controllable, human-editable). The `SkillLibrary` maintains an in-memory cache and supports both keyword-based and FAISS embedding-based retrieval.

---

## Environment Contract

Every environment implements `SEAEnv`:

```python
class SEAEnv(ABC):
    def reset(self, *, seed=None, task_id=None) -> tuple[Observation, dict]: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...
    def get_task_ids(self) -> list[str]: ...
    def close(self) -> None: ...
```

| Benchmark | Task Types | task_id Support |
|-----------|------------|-----------------|
| TextCraft | Minecraft crafting recipes | `seed_0`..`seed_N` |
| ALFWorld | 6 household tasks (pick, clean, heat, cool, examine, pick_two) | Per-game file selection |
| WebShop | Web shopping sessions | Session ID |

ALFWorld supports per-game task_id selection via game file indexing — `get_task_ids()` returns real selectable IDs derived from game file paths.

---

## Evolution Methods

### Evolver Interface

```python
class Evolver(Checkpointable):
    def evolve(self, agent, target: Evolvable, trajectories, metrics, **kw): ...
    def requires_trajectories(self) -> bool: ...
```

### Available Methods

| Method | Target | How It Works |
|--------|--------|-------------|
| **SFTEvolver** | `LoRATarget (Path)` | Supervised fine-tuning on successful trajectories. Manual chat-template tokenization with assistant-only label masking. |
| **RLEvolver** (REINFORCE) | `LoRATarget (Path)` | Offline policy gradient: computes discounted returns G_t per step, trains with `-log_prob × advantage` loss. |
| **RLEvolver** (DPO) | `LoRATarget (Path)` | Direct preference optimization from trajectory pairs. |
| **ICLEvolver** | `Memory (list[dict])` | Reflexion: generates verbal reflections on failures, curates exemplars from successes, writes to memory. |
| **ExpeLEvolver** | `Memory (list[dict])` | Extracts IF→THEN→BECAUSE semantic rules from trajectory analysis. |
| **PromptEvolver** | `PromptTarget (str)` | LLM-based prompt mutation and selection. |

### External Model Support

SFTEvolver and RLEvolver accept optional `model=` and `tokenizer=` kwargs. When provided, the evolver reuses the caller's model without loading/freeing, enabling efficient single-model train/eval loops.

---

## Data Collection

Three collection modes for different scenarios:

| Method | Use Case | Concurrency |
|--------|----------|-------------|
| `collect()` | Serial collection (memory/skill evolution needs shared state) | None |
| `collect_parallel()` | **API high-concurrency collection** (SFT/RL training data) | ThreadPool, N workers, each with own agent+env |
| `collect_subprocess()` | Local env parallel collection (ALFWorld, non-thread-safe) | Subprocess + JSONL atomic writes |

### `collect_parallel()` Design

- Factory pattern: `agent_factory` and `env_factory` create independent instances per worker
- Real-time target tracking: workers loop until exact count is met (no oversample)
- `only_successful` mode: discards failures, keeps collecting until N successes
- Workers stop via `threading.Event` when target is reached

---

## LLM Backends

| Backend | Purpose | LoRA Support |
|---------|---------|-------------|
| `APIBackend` | Remote inference via OpenAI-compatible API | No |
| `VLLMBackend` | Local inference with vLLM engine | Hot-swap via `load_lora/unload_lora` |
| `HFTrainingBackend` | Local training with HuggingFace Transformers + PEFT | LoRA training + adapter save/load |

### Typical Workflow

```
Phase A: APIBackend + collect_parallel() → high-concurrency data collection (no GPU)
Phase B: HFTrainingBackend → LoRA training on GPU
Phase C: VLLMBackend or HFTrainingBackend → evaluation with trained adapter
```

---

## Evolution Pipeline

`EvolutionPipeline` orchestrates the full loop:

```python
pipeline = EvolutionPipeline(
    agent=agent,
    envs=[textcraft_env],
    evolvers=[(sft_evolver, "lora_target")],
    evaluator=evaluator,
    metrics=metrics,
    config=EvolutionConfig(num_iterations=100, ...),
    extra_targets={"lora_target": lora_target},
)
pipeline.run()
```

Each iteration: collect trajectories → evolve targets → periodic evaluation → periodic checkpoint.

---

## Metrics and Evaluation

### Evaluator

Runs episodes across environments, aggregates success rate, average reward, and per-task breakdown.

### MetricsTracker + Reporters

Dispatches metrics to one or more reporters:

- `ConsoleReporter` — Rich table output to terminal
- `TensorBoardReporter` — TensorBoard scalar logging
- `WandBReporter` — Weights & Biases integration

---

## Configuration

YAML configs with OmegaConf, supporting dot-notation overrides:

```bash
python scripts/run_evolution.py \
    --config configs/textcraft_sft.yaml \
    evolution.pipeline.num_iterations=50 \
    agent.brain.model=Qwen/Qwen2.5-7B-Instruct
```

---

## Key Design Decisions

1. **Evolvable[T] is generic** — The same pipeline evolves LoRA weights (T=Path), prompts (T=str), memory (T=list[dict]), and skills (T=list[dict]) uniformly.

2. **Skills use SKILL.md** — File-system based, human-readable markdown files with YAML frontmatter, following the 2025 Progressive Disclosure standard.

3. **Collection is decoupled from training** — API-based collection (fast, parallel) feeds into local GPU training. Data is cached to avoid redundant API calls.

4. **Single-GPU train/eval** — Tutorials alternate loading model for eval and training on the same GPU, freeing memory between stages.

5. **Environments support task_id selection** — ALFWorld indexes game files for per-game selection. TextCraft uses seed-based task selection. This enables reproducible evaluation.

6. **REINFORCE replaces GRPO** — Offline trajectory-level policy gradient using pre-collected rewards, not online generation. Works with any trajectory data.
