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
  <a href="../README.md">English</a>
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
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd SEA && pip install -r requirements.txt
```

### 2. 最小示例

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

# 3. 定义环境
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

# 4. 收集 → 进化 → 评测
collector = TrajectoryCollector()
trajectories = collector.collect(agent, [env], n=10)

evolver = ICLEvolver(max_reflections_per_step=3, max_exemplars=5)
metrics = MetricsTracker()
evolver.evolve(agent, agent.evolvable_components()["memory"], trajectories, metrics)

results = Evaluator(num_episodes_per_env=10).evaluate(agent, [env])
print(f"成功率: {results.success_rate:.0%}")
```

### 3. 使用 YAML 配置运行完整进化循环

```bash
python scripts/run_evolution.py --config examples/lora_sft_textcraft/config.yaml
```

---

## 平台架构

### 核心协议

```python
class Evolvable(Checkpointable, Generic[T]):
    """可进化的组件。T = 状态类型。"""
    def get_evolvable_state(self) -> T: ...
    def set_evolvable_state(self, state: T) -> None: ...

class Evolver(Checkpointable):
    def evolve(self, agent, target: Evolvable, trajectories, metrics, **kwargs) -> None: ...
```

**任何进化方法都可以作用于任何兼容的进化对象：**

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
├── SkillLibrary     — FAISS 索引，文本 / 代码 / 组合技能（Evolvable）
└── ToolRegistry     — calculator, json_parser 及自定义工具
```

### LLM 后端 —— 推理/训练分离

| 组件 | GPU | 用途 |
|---|---|---|
| `VLLMBackend` | GPU 0 | 快速推理，通过 `LoRARequest` 实现 LoRA 热切换 |
| `HFTrainingBackend` | GPU 1 | PEFT + TRL 进行 SFT / RL 训练（支持自定义 callbacks） |

### 基准环境

| 基准 | 安装方式 | 任务类型 | 状态 |
|---|---|---|---|
| **TextCraft** | `pip install textcraft` | Minecraft 合成 | ✅ 已验证 |
| **ALFWorld** | `pip install alfworld` + `alfworld-download` | 家务机器人（6种任务） | ✅ 已验证 |
| **WebShop** | `git clone` + `setup.sh` | 电商导航 | 适配器就绪 |

ALFWorld 支持**任务类型过滤**（pick, clean, heat, cool, examine, pick_two），适用于多任务和持续学习实验。

---

## 扩展指南

### 添加新的进化方法

```python
@EVOLVER_REGISTRY.register("my_method")
class MyEvolver(Evolver):
    def evolve(self, agent, target, trajectories, metrics, **kwargs):
        state = target.get_evolvable_state()
        # ... 你的进化逻辑 ...
        target.set_evolvable_state(new_state)
```

### 添加新的环境

```python
@ENV_REGISTRY.register("my_env")
class MyEnv(SEAEnv):
    def reset(self, *, seed=None, task_id=None):
        return Observation(text="开始"), {"task_id": task_id, "task_type": "my_type"}
    def step(self, action):
        return Observation(text="结果"), reward, done, False, {"success": done}
```

### 添加组合技能

```python
from sea.agent.skills.code_skill import ComposedSkill

skill = ComposedSkill(
    name="clean_and_place",
    description="清洁物品并放置到指定位置",
    composition_plan="navigate(obj) → pick(obj) → clean(obj) → navigate(dest) → put(obj, dest)",
    sub_skills=["navigate", "pick_up", "clean", "put_on"],
)
agent.skill_library.add_skill(skill)
```

> 完整的 Claude Code 提示模板请参阅 [Prompt Templates](PROMPTS.md)。

---

## 项目结构

```
sea/
├── core/             # Evolvable[T]、Checkpointable、类型定义、组件注册
├── agent/            # SEAAgent、Brain、Planner、Memory、Skills、Tools
├── llm/              # vLLM（LoRA 热切换）、API 后端、HF 训练后端
├── evolution/
│   ├── targets/      # LoRA（多适配器）、Prompt、Memory、Skill 进化对象
│   ├── methods/      # SFT、RL (GRPO/DPO)、ICL (Reflexion)、Prompt、ExpeL
│   ├── data/         # 轨迹收集（任务类型过滤）、奖励函数、数据集转换
│   └── pipeline.py   # EvolutionPipeline: 收集 → 进化 → 评测 → 重复
├── env/              # SEAEnv 接口 + TextCraft、ALFWorld、WebShop 适配器
├── metrics/          # MetricsTracker、Evaluator（固定种子）、reporters
└── utils/            # 配置（OmegaConf）、日志、序列化
```

---

## 文档

| 文档 | 说明 |
|---|---|
| **[Prompt 模板](PROMPTS.md)** | Claude Code vibe coding 模板 —— 分钟级实现想法 |
| **[SFT 教程](TUTORIAL_SFT.md)** | TextCraft 上的 LoRA SFT（本地 GPU） |
| **[RL 教程](TUTORIAL_RL.md)** | TextCraft 上的 GRPO + 环境奖励 |
| **[Memory 教程](TUTORIAL_MEMORY.md)** | TextCraft 上的 ICL/Reflexion 记忆进化 |
| **[Skill 教程](TUTORIAL_SKILLS.md)** | TextCraft 上的自定义技能进化 |
| **[端到端教程](TUTORIAL.md)** | 第一个进化实验的完整演练 |

---

## 脚本

| 脚本 | 说明 |
|---|---|
| `python scripts/run_evolution.py --config <yaml>` | 运行完整进化循环 |
| `python scripts/run_eval.py --config <yaml> --checkpoint <path>` | 评测某个检查点 |
| `python scripts/serve_model.py --model <name>` | 启动 vLLM OpenAI 兼容服务 |
| `python scripts/collect_trajectories.py --config <yaml> -n 100` | 仅收集轨迹（不训练） |

---

## 示例

| 示例 | 方法 | 进化对象 | 环境 |
|---|---|---|---|
| `examples/sft_textcraft/` | SFT (LoRA) | LM 权重 | TextCraft |
| `examples/rl_textcraft/` | GRPO (LoRA) | LM 权重 | TextCraft |
| `examples/memory_textcraft/` | ICL (Reflexion) | 记忆 | TextCraft |
| `examples/skill_textcraft/` | 自定义 Evolver | 技能库 | TextCraft |
| `examples/expel_textcraft/` | ExpeL | 记忆（规则） | TextCraft |

---

## 引用

```bibtex
@software{sea2025,
  title  = {SEA: Self-Evolving Agent Platform},
  author = {Ivo Wang},
  year   = {2025},
  url    = {https://github.com/ivowang/SEA},
}
```

## 许可证

MIT
