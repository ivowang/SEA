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
  <a href="#中文文档">中文文档</a>
</p>

---

## Why SEA?

Existing agent frameworks focus on *using* agents. SEA focuses on **evolving** them.

SEA decouples **what evolves** (LoRA weights, prompts, memory, skills) from **how it evolves** (SFT, RL, in-context learning, prompt optimization), letting researchers freely combine evolution methods and targets in a single config file.

```
┌─────────────────────────────────────────────────────┐
│              EvolutionPipeline                       │
│        collect → evolve → evaluate → repeat         │
├──────────┬──────────┬──────────┬────────────────────┤
│ Evolvers │ Targets  │ Metrics  │ TrajectoryBuffer   │
│ SFT / RL │ LoRA /   │ Tracker +│ Collect / Filter / │
│ ICL /... │ Prompt / │ Evaluator│ Sample             │
│          │ Memory / │          │                    │
│          │ Skill    │          │                    │
├──────────┴──────────┴──────────┴────────────────────┤
│                    SEAAgent                          │
│  ┌──────┐ ┌───────┐ ┌───────┐ ┌──────┐ ┌────────┐  │
│  │Brain │ │Memory │ │Planner│ │Skills│ │ Tools  │  │
│  │(LLM) │ │Epi/Sem│ │ReAct/ │ │Library│ │Registry│  │
│  │      │ │/Work  │ │LATS   │ │      │ │        │  │
│  └──┬───┘ └───────┘ └───────┘ └──────┘ └────────┘  │
├─────┴───────────────────────────────────────────────┤
│  LLM Backend                                        │
│  GPU 0: vLLM inference + LoRA hot-swap              │
│  GPU 1: HF Trainer (PEFT + TRL) for SFT/RL         │
├─────────────────────────────────────────────────────┤
│  Environment Layer: SEAEnv (Gymnasium-style)        │
│  TextCraft │ ALFWorld │ WebShop │ Custom …          │
└─────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---|---|
| **Evolution-first design** | `Evolvable[T]` protocol makes any component an evolution target |
| **Method-target decoupling** | Any `Evolver` works with any compatible `Evolvable` target |
| **LoRA hot-swap** | Train a new adapter on GPU 1, hot-swap into vLLM on GPU 0 — zero downtime |
| **4 built-in evolvers** | SFT, RL (GRPO/DPO), ICL (Reflexion), Prompt (SCOPE/EvoPrompt) |
| **4 evolution targets** | LM parameters (LoRA), system prompt, memory, skill library |
| **3 benchmarks** | TextCraft, ALFWorld, WebShop — all widely used in agent evolution papers |
| **Plug-in everything** | `@REGISTRY.register("name")` + one YAML line to use your custom component |
| **Real-time metrics** | Console, TensorBoard, and W&B reporters track evolution progress |

---

## Quick Start

### 1. Install

```bash
conda create -n sea python=3.11 -y
conda activate sea
cd SEA
pip install -e .
```

### 2. Minimal Example — evolve an agent in 40 lines

The following script creates an agent, runs it in a custom environment, collects trajectories, evolves its memory via ICL (Reflexion), and evaluates the improvement. **No GPU required** — it uses an OpenAI-compatible API as the LLM backend.

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

# 4. Collect trajectories
collector = TrajectoryCollector()
trajectories = collector.collect(agent, [env], n=10)

# 5. Evolve via ICL (Reflexion — no GPU needed)
evolver = ICLEvolver(max_reflections_per_step=3, max_exemplars=5)
metrics = MetricsTracker()
memory_target = agent.evolvable_components()["memory"]
evolver.evolve(agent, memory_target, trajectories, metrics)

# 6. Evaluate improvement
evaluator = Evaluator(num_episodes_per_env=10)
results = evaluator.evaluate(agent, [env])
print(f"Success rate: {results.success_rate:.0%}")
```

### 3. Full evolution loop with YAML config

For GPU-based experiments (SFT / RL), use the config-driven pipeline:

```bash
# Start vLLM server on GPU 0
python scripts/serve_model.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Run SFT evolution on TextCraft
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

The config file controls everything:

```yaml
# examples/lora_sft_textcraft/config.yaml
agent:
  brain:
    backend: vllm
    model: Qwen/Qwen2.5-1.5B-Instruct
    enable_lora: true
    device: "cuda:0"
  memory: working
  planner: react

env:
  name: textcraft
  max_steps_val: 30

evolution:
  pipeline:
    num_iterations: 20
    traj_per_iter: 16
    eval_every: 5
  evolvers:
    - method: sft           # SFT / rl / icl / prompt
      target: brain         # brain / memory / skill_library
      device: "cuda:1"
      learning_rate: 2.0e-5
      num_epochs: 2

metrics:
  reporters: [console, tensorboard]
```

---

## Architecture

### Core Protocols

Everything in SEA is built on two protocols:

```python
class Checkpointable(ABC):
    """Save / load state."""
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...

class Evolvable(Checkpointable, Generic[T]):
    """A component that can be evolved. T = state type."""
    def get_evolvable_state(self) -> T: ...
    def set_evolvable_state(self, state: T) -> None: ...
```

The `Evolver` ABC operates on any `Evolvable`:

```python
class Evolver(Checkpointable):
    def evolve(self, agent, target: Evolvable, trajectories, metrics) -> None: ...
```

This separation means **any evolver can work with any target**:

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
├── SkillLibrary     — FAISS-indexed, code or text skills (Evolvable)
└── ToolRegistry     — calculator, finish, json_parser, custom tools
```

### LLM Backend — Inference/Training Split

| Component | GPU | Role |
|---|---|---|
| `VLLMBackend` | GPU 0 | Fast inference, LoRA hot-swap via `LoRARequest` |
| `HFTrainingBackend` | GPU 1 | PEFT + TRL for SFT / RL training |

After training produces a new LoRA checkpoint on disk, the vLLM backend hot-swaps to it without restart.

### Environments

`SEAEnv` mirrors Gymnasium but uses text-centric types:

```python
class SEAEnv(ABC):
    def reset(self, *, seed=None, task_id=None) -> tuple[Observation, dict]: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...
    def get_task_ids(self) -> list[str]: ...
```

Built-in benchmarks: **TextCraft** (crafting), **ALFWorld** (household), **WebShop** (e-commerce).

Adapting your own environment:

```python
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation
from sea.core.registry import ENV_REGISTRY

@ENV_REGISTRY.register("my_env")
class MyEnv(SEAEnv):
    @property
    def name(self) -> str:
        return "my_env"

    def reset(self, *, seed=None, task_id=None):
        return Observation(text="Start"), {"task_id": task_id}

    def step(self, action):
        reward = 1.0 if "correct" in action.text else 0.0
        done = reward > 0
        return Observation(text="Result"), reward, done, False, {}

    def get_task_ids(self):
        return ["task_0", "task_1"]
```

Then reference it in your config: `env: { name: my_env }`.

---

## Extending SEA

### Add a new Evolution Method (~50 lines)

```python
from sea.core.registry import EVOLVER_REGISTRY
from sea.evolution.base import Evolver

@EVOLVER_REGISTRY.register("my_method")
class MyEvolver(Evolver):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def requires_trajectories(self) -> bool:
        return True

    def evolve(self, agent, target, trajectories, metrics):
        # Your evolution logic here
        # Read:  state = target.get_evolvable_state()
        # Write: target.set_evolvable_state(new_state)
        successful = [t for t in trajectories if t.success]
        metrics.log({"my_method/num_successful": len(successful)})
```

Use it in config:

```yaml
evolution:
  evolvers:
    - method: my_method
      target: memory
      temperature: 0.8
```

### Add a new Evolution Target

```python
from sea.core.base import Evolvable

class MyTarget(Evolvable[dict]):
    def get_evolvable_state(self) -> dict:
        return {"key": "value"}

    def set_evolvable_state(self, state: dict) -> None:
        # Apply new state
        ...

    # Also implement: evolution_metadata, save_checkpoint, load_checkpoint, state_dict
```

### Add a new Tool

```python
from sea.agent.tools.base import Tool, ToolResult
from sea.core.registry import TOOL_REGISTRY

@TOOL_REGISTRY.register("web_search")
class WebSearchTool(Tool):
    @property
    def name(self) -> str: return "web_search"

    @property
    def description(self) -> str: return "Search the web for information."

    def execute(self, query: str = "", **kw) -> ToolResult:
        # Your search logic
        return ToolResult(output=f"Results for: {query}")
```

---

## Project Structure

```
sea/
├── core/             # Evolvable[T], Checkpointable, types, registry
├── agent/            # SEAAgent, Brain, Planner, Memory, Skills, Tools
├── llm/              # vLLM (LoRA hot-swap), API backend, HF training backend
├── evolution/
│   ├── targets/      # LoRA, Prompt, Memory, Skill targets
│   ├── methods/      # SFT, RL (GRPO/DPO), ICL (Reflexion), Prompt optimizer
│   ├── data/         # Trajectory collection, reward functions, dataset conversion
│   └── pipeline.py   # EvolutionPipeline: collect → evolve → evaluate → repeat
├── env/              # SEAEnv ABC + TextCraft, ALFWorld, WebShop adapters
├── metrics/          # MetricsTracker, Evaluator, Console/TensorBoard/W&B reporters
└── utils/            # Config (OmegaConf), logging, serialization
```

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

| Example | Method | Target | Environment | Config |
|---|---|---|---|---|
| LoRA SFT | SFT | LM weights (LoRA) | TextCraft | `examples/lora_sft_textcraft/config.yaml` |
| RL GRPO | GRPO | LM weights (LoRA) | ALFWorld | `examples/rl_grpo_alfworld/config.yaml` |
| ICL Reflexion | ICL | Memory | WebShop | `examples/icl_reflexion_webshop/config.yaml` |

---

## Citation

If you use SEA in your research, please cite:

```bibtex
@software{sea2025,
  title  = {SEA: Self-Evolving Agent Platform},
  author = {Ivo Wang},
  year   = {2025},
  url    = {https://github.com/ivowang/SEA},
}
```

---

## License

MIT

---

---

<a id="中文文档"></a>

# 中文文档

## SEA：自进化智能体研究平台

SEA 是一个面向**自进化智能体（Self-Evolving Agent）**研究的通用科研平台。研究者可以在此平台上快速实验各种进化方法（SFT、RL、上下文学习、Prompt优化等），作用于不同的进化对象（LoRA权重、Prompt、记忆、技能库），并通过标准化的环境接口和实时评测系统追踪进化效果。

### 核心理念

SEA 将 **"进化什么"** 和 **"怎么进化"** 彻底解耦：

- **进化对象** (`Evolvable[T]`)：LoRA适配器、系统Prompt、记忆库、技能库 —— 每个都有自己的状态类型 `T`
- **进化方法** (`Evolver`)：SFT、GRPO/DPO、Reflexion式上下文学习、Prompt优化 —— 每个都可作用于任何兼容的对象
- **核心循环**：收集轨迹 → 进化对象 → 评测表现 → 重复

一个新的进化方法只需 ~50 行代码 + 一行 YAML 配置即可接入平台。

### 快速开始

#### 安装

```bash
conda create -n sea python=3.11 -y
conda activate sea
cd SEA
pip install -e .
```

#### 最小示例 —— 40 行代码完成一次 Agent 进化

以下代码演示了完整的进化流程：创建 Agent → 定义环境 → 收集轨迹 → ICL 进化 → 评测。**无需 GPU**，使用任意 OpenAI 兼容接口即可。

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

# 1. LLM 后端 —— 任意 OpenAI 兼容接口
backend = APIBackend(model="Qwen/Qwen2.5-7B-Instruct",
                     base_url="http://localhost:8000/v1")

# 2. 组装 Agent
agent = SEAAgent(
    brain=LLMBrain(backend=backend),
    memory=EpisodicMemory(),
    planner=ReActPlanner(),
)

# 3. 定义一个简单环境
def reset_fn(**kw):
    return Observation(text="你在一个房间里，找到钥匙并打开门。"), \
           {"task_id": "room_escape", "task_description": "找到钥匙，打开门。"}

def step_fn(action):
    if "key" in action.text.lower() and "door" in action.text.lower():
        return Observation(text="你逃出来了！"), 1.0, True, False, {"success": True}
    if "key" in action.text.lower():
        return Observation(text="你找到了一把钥匙。"), 0.1, False, False, {}
    return Observation(text="什么都没发生。"), 0.0, False, False, {}

env = FunctionEnv(name="room_escape", reset_fn=reset_fn, step_fn=step_fn,
                  task_ids=["room_escape"], max_steps_val=10)

# 4. 收集轨迹
collector = TrajectoryCollector()
trajectories = collector.collect(agent, [env], n=10)

# 5. ICL 进化（Reflexion —— 无需 GPU）
evolver = ICLEvolver(max_reflections_per_step=3, max_exemplars=5)
metrics = MetricsTracker()
memory_target = agent.evolvable_components()["memory"]
evolver.evolve(agent, memory_target, trajectories, metrics)

# 6. 评测改进效果
evaluator = Evaluator(num_episodes_per_env=10)
results = evaluator.evaluate(agent, [env])
print(f"成功率: {results.success_rate:.0%}")
```

#### 使用 YAML 配置运行完整进化循环

对于需要 GPU 的实验（SFT / RL），使用配置文件驱动：

```bash
# 在 GPU 0 启动 vLLM 推理服务
python scripts/serve_model.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 运行 SFT 进化实验
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

### 平台架构

```
sea/
├── core/             # Evolvable[T]、Checkpointable 协议、数据类型、组件注册
├── agent/            # SEAAgent、Brain、Planner、Memory、Skills、Tools
├── llm/              # vLLM（LoRA热切换）、API后端、HuggingFace训练后端
├── evolution/
│   ├── targets/      # 进化对象：LoRA、Prompt、Memory、Skill
│   ├── methods/      # 进化方法：SFT、RL (GRPO/DPO)、ICL (Reflexion)、Prompt优化
│   ├── data/         # 轨迹收集、奖励函数、数据集转换
│   └── pipeline.py   # EvolutionPipeline: 收集 → 进化 → 评测 → 重复
├── env/              # SEAEnv 接口 + TextCraft、ALFWorld、WebShop 适配器
├── metrics/          # MetricsTracker、Evaluator、Console/TensorBoard/W&B
└── utils/            # 配置（OmegaConf）、日志、序列化
```

### GPU 分配策略（2卡配置）

| 组件 | GPU | 用途 |
|---|---|---|
| `VLLMBackend` | GPU 0 | 快速推理，通过 `LoRARequest` 实现 LoRA 热切换 |
| `HFTrainingBackend` | GPU 1 | PEFT + TRL 进行 SFT / RL 训练 |

训练完成后将新的 LoRA 检查点保存到磁盘，vLLM 后端热切换至新适配器，无需重启。

### 如何接入自己的环境

实现 `SEAEnv` 接口即可：

```python
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation
from sea.core.registry import ENV_REGISTRY

@ENV_REGISTRY.register("my_env")
class MyEnv(SEAEnv):
    @property
    def name(self): return "my_env"

    def reset(self, *, seed=None, task_id=None):
        return Observation(text="初始观察"), {"task_id": task_id}

    def step(self, action):
        reward = 1.0 if "正确" in action.text else 0.0
        done = reward > 0
        return Observation(text="结果"), reward, done, False, {}

    def get_task_ids(self):
        return ["task_0", "task_1"]
```

然后在配置中引用：`env: { name: my_env }`。

### 如何实现自己的进化方法

继承 `Evolver`，实现 `evolve()` 方法，用 `@EVOLVER_REGISTRY.register()` 注册：

```python
from sea.core.registry import EVOLVER_REGISTRY
from sea.evolution.base import Evolver

@EVOLVER_REGISTRY.register("my_method")
class MyEvolver(Evolver):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def requires_trajectories(self):
        return True

    def evolve(self, agent, target, trajectories, metrics):
        # 读取当前状态
        state = target.get_evolvable_state()
        # ... 你的进化逻辑 ...
        # 写回新状态
        target.set_evolvable_state(new_state)
        metrics.log({"my_method/score": score})
```

在配置中使用：

```yaml
evolution:
  evolvers:
    - method: my_method
      target: memory
      temperature: 0.8
```

### 内置组件一览

| 类别 | 组件 |
|---|---|
| **进化方法** | SFT (`sft`)、RL (`rl`, 支持 GRPO/DPO)、ICL (`icl`, Reflexion式)、Prompt优化 (`prompt`) |
| **进化对象** | LoRA权重 (`LoRATarget`)、Prompt (`PromptTarget`)、记忆 (`MemoryTarget`)、技能库 (`SkillTarget`) |
| **Agent组件** | LLMBrain、ReActPlanner、EpisodicMemory、SemanticMemory、WorkingMemory、SkillLibrary、ToolRegistry |
| **LLM后端** | vLLM（本地推理+LoRA热切换）、vLLM Server（远程推理）、API（OpenAI兼容） |
| **环境** | TextCraft（合成任务）、ALFWorld（家务机器人）、WebShop（电商导航） |
| **评测** | MetricsTracker、Evaluator、Console/TensorBoard/W&B reporter |

### 引用

```bibtex
@software{sea2025,
  title  = {SEA: Self-Evolving Agent Platform},
  author = {Ivo Wang},
  year   = {2025},
  url    = {https://github.com/ivowang/SEA},
}
```
