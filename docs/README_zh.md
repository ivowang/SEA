<p align="center">
  <img src="../assets/logo.svg" alt="SEA Logo" width="280"/>
</p>

<h1 align="center">SEA：自进化智能体研究平台</h1>

<p align="center">
  <b>面向自进化智能体（Self-Evolving Agent）研究的通用科研平台</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> &bull;
  <a href="#平台架构">平台架构</a> &bull;
  <a href="#扩展指南">扩展指南</a> &bull;
  <a href="../README.md">English</a> &bull;
  <a href="TUTORIAL.md">实验教程</a>
</p>

---

## 为什么选择 SEA？

现有的 Agent 框架专注于**使用** Agent。SEA 专注于**进化**它们。

SEA 将 **"进化什么"**（LoRA 权重、Prompt、记忆、技能）和 **"怎么进化"**（SFT、RL、上下文学习、Prompt优化）彻底解耦，研究者可以在一个配置文件中自由组合进化方法和进化对象。

```
┌─────────────────────────────────────────────────────┐
│              EvolutionPipeline                      │
│        收集轨迹 → 进化对象 → 评测表现 → 重复         │
├──────────┬──────────┬──────────┬────────────────────┤
│ 进化方法  │ 进化对象  │  评测    │  轨迹缓冲          │
│ SFT / RL │ LoRA /   │ Tracker +│ 收集 / 过滤 /      │
│ ICL /... │ Prompt / │ Evaluator│ 采样               │
│          │ Memory / │          │                    │
│          │ Skill    │          │                    │
├──────────┴──────────┴──────────┴────────────────────┤
│                    SEAAgent                         │
│  ┌──────┐ ┌───────┐ ┌───────┐ ┌──────┐ ┌────────┐   │
│  │LLM作 │ │记忆   │ │规划器 │ │技能  │ │工具    │   │
│  │为大脑│ │Epi/Sem│ │ReAct/ │ │技能库│ │注册表  │   │
│  │      │ │/Work  │ │LATS   │ │      │ │        │   │
│  └──┬───┘ └───────┘ └───────┘ └──────┘ └────────┘   │
├─────┴───────────────────────────────────────────────┤
│  LLM 后端                                           │
│  GPU 0: vLLM 推理 + LoRA 热切换                      │
│  GPU 1: HF Trainer (PEFT + TRL) SFT/RL 训练          │
├─────────────────────────────────────────────────────┤
│  环境层: SEAEnv (Gymnasium 风格)                      │
│  TextCraft │ ALFWorld │ WebShop │ 自定义 …            │
└─────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 安装

```bash
conda create -n sea python=3.11 -y
conda activate sea

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

cd SEA
pip install -e .
```

### 2. 最小示例

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

### 3. 使用 YAML 配置运行完整进化循环

对于需要 GPU 的实验（SFT / RL），使用配置文件驱动：

```bash
# 在 GPU 0 启动 vLLM 推理服务
python scripts/serve_model.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 运行 SFT 进化实验
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

配置文件控制所有参数：

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

> 完整的端到端实验教程（含 GPU SFT 训练、LoRA 热切换、实测成功率从 67% 提升至 100%），请参阅 **[实验教程](TUTORIAL.md)**。

---

## 平台架构

### 核心协议

SEA 的一切建立在两个协议之上：

```python
class Checkpointable(ABC):
    """保存 / 加载状态"""
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...

class Evolvable(Checkpointable, Generic[T]):
    """可进化的组件。T = 状态类型。"""
    def get_evolvable_state(self) -> T: ...
    def set_evolvable_state(self, state: T) -> None: ...
```

`Evolver` 抽象类作用于任意 `Evolvable`：

```python
class Evolver(Checkpointable):
    def evolve(self, agent, target: Evolvable, trajectories, metrics) -> None: ...
```

这意味着**任何进化方法都可以作用于任何兼容的进化对象**：

| | LoRA (`Path`) | Prompt (`str`) | Memory (`list[dict]`) | Skills (`list[dict]`) |
|---|:---:|:---:|:---:|:---:|
| **SFT** | ✅ | — | — | — |
| **RL (GRPO/DPO)** | ✅ | — | — | — |
| **ICL (Reflexion)** | — | — | ✅ | ✅ |
| **Prompt 优化** | — | ✅ | — | — |

### Agent 组件

```
SEAAgent
├── LLMBrain        — 封装 LLM 后端，管理 LoRA + 系统 Prompt（Evolvable）
├── Memory           — episodic / semantic (FAISS) / working（Evolvable）
├── Planner          — ReAct（默认），可扩展至 LATS / ToT
├── SkillLibrary     — FAISS 索引，代码或文本技能（Evolvable）
└── ToolRegistry     — calculator, finish, json_parser 及自定义工具
```

### LLM 后端 —— 推理/训练分离

| 组件 | GPU | 用途 |
|---|---|---|
| `VLLMBackend` | GPU 0 | 快速推理，通过 `LoRARequest` 实现 LoRA 热切换 |
| `HFTrainingBackend` | GPU 1 | PEFT + TRL 进行 SFT / RL 训练 |

训练完成后将新的 LoRA 检查点保存到磁盘，vLLM 后端热切换至新适配器，无需重启。

### 环境

`SEAEnv` 遵循 Gymnasium 风格，但使用文本类型：

```python
class SEAEnv(ABC):
    def reset(self, *, seed=None, task_id=None) -> tuple[Observation, dict]: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...
    def get_task_ids(self) -> list[str]: ...
```

内置基准环境：**TextCraft**（合成任务）、**ALFWorld**（家务机器人）、**WebShop**（电商导航）。

---

## 扩展指南

### 添加新的进化方法

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
        # 你的进化逻辑
        # 读取:  state = target.get_evolvable_state()
        # 写回:  target.set_evolvable_state(new_state)
        successful = [t for t in trajectories if t.success]
        metrics.log({"my_method/num_successful": len(successful)})
```

在配置中使用：

```yaml
evolution:
  evolvers:
    - method: my_method
      target: memory
      temperature: 0.8
```

### 添加新的环境

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
        return Observation(text="初始观察"), {"task_id": task_id}

    def step(self, action):
        reward = 1.0 if "正确" in action.text else 0.0
        done = reward > 0
        return Observation(text="结果"), reward, done, False, {}

    def get_task_ids(self):
        return ["task_0", "task_1"]
```

然后在配置中引用：`env: { name: my_env }`。

### 添加新的进化对象

```python
from sea.core.base import Evolvable

class MyTarget(Evolvable[dict]):
    def get_evolvable_state(self) -> dict:
        return {"key": "value"}

    def set_evolvable_state(self, state: dict) -> None:
        ...  # 应用新状态

    # 还需实现: evolution_metadata, save_checkpoint, load_checkpoint, state_dict
```

### 添加新的工具

```python
from sea.agent.tools.base import Tool, ToolResult
from sea.core.registry import TOOL_REGISTRY

@TOOL_REGISTRY.register("web_search")
class WebSearchTool(Tool):
    @property
    def name(self) -> str: return "web_search"

    @property
    def description(self) -> str: return "搜索网络获取信息。"

    def execute(self, query: str = "", **kw) -> ToolResult:
        return ToolResult(output=f"搜索结果: {query}")
```

---

## 项目结构

```
sea/
├── core/             # Evolvable[T]、Checkpointable、类型定义、组件注册
├── agent/            # SEAAgent、Brain、Planner、Memory、Skills、Tools
├── llm/              # vLLM（LoRA 热切换）、API 后端、HF 训练后端
├── evolution/
│   ├── targets/      # LoRA、Prompt、Memory、Skill 进化对象
│   ├── methods/      # SFT、RL (GRPO/DPO)、ICL (Reflexion)、Prompt 优化
│   ├── data/         # 轨迹收集、奖励函数、数据集转换
│   └── pipeline.py   # EvolutionPipeline: 收集 → 进化 → 评测 → 重复
├── env/              # SEAEnv 接口 + TextCraft、ALFWorld、WebShop 适配器
├── metrics/          # MetricsTracker、Evaluator、Console/TensorBoard/W&B
└── utils/            # 配置（OmegaConf）、日志、序列化
```

---

## 脚本与示例

| 脚本 | 说明 |
|---|---|
| `python scripts/run_evolution.py --config <yaml>` | 运行完整进化循环 |
| `python scripts/run_eval.py --config <yaml> --checkpoint <path>` | 评测某个检查点 |
| `python scripts/serve_model.py --model <name>` | 启动 vLLM OpenAI 兼容服务 |
| `python scripts/collect_trajectories.py --config <yaml> -n 100` | 仅收集轨迹（不训练） |

---

| 示例 | 方法 | 进化对象 | 环境 | 配置 |
|---|---|---|---|---|
| LoRA SFT | SFT | LM 权重 (LoRA) | TextCraft | `examples/lora_sft_textcraft/config.yaml` |
| RL GRPO | GRPO | LM 权重 (LoRA) | ALFWorld | `examples/rl_grpo_alfworld/config.yaml` |
| ICL Reflexion | ICL | 记忆 | WebShop | `examples/icl_reflexion_webshop/config.yaml` |
| **端到端 Demo** | SFT | LM 权重 (LoRA) | Simple Tasks | `examples/e2e_demo/run.py` |
