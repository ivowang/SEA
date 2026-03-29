# Vibe Coding Prompt Templates for SEA

These prompt templates are designed for researchers using **Claude Code** (or similar AI coding assistants) to quickly implement new ideas on the SEA platform. Copy the template, fill in the `[BRACKETS]`, and paste into Claude Code.

---

## Template 1: Implement a New Evolution Method

```
I want to implement a new evolution method called [METHOD_NAME] on the SEA platform
(github.com/ivowang/SEA).

**What it does**: [DESCRIPTION OF THE METHOD]

**What it evolves**: [TARGET: LoRA weights / prompt / memory / skills]

**How it works**:
[STEP-BY-STEP DESCRIPTION OF THE ALGORITHM]

**Reference paper**: [PAPER TITLE AND URL, IF ANY]

Please implement this as a new Evolver class in sea/evolution/methods/[NAME].py.
Follow the existing pattern:
- Inherit from `sea.evolution.base.Evolver`
- Register with `@EVOLVER_REGISTRY.register("[NAME]")`
- Implement `evolve(self, agent, target, trajectories, metrics, **kwargs)`
- Implement `requires_trajectories()`

Key files to reference:
- sea/evolution/base.py — Evolver ABC
- sea/evolution/methods/sft.py — SFT example
- sea/evolution/methods/icl.py — ICL example
- sea/core/types.py — Trajectory, Step, Action types
- sea/agent/agent.py — SEAAgent interface

Then create a runnable example in examples/[NAME]/run.py that demonstrates
the method on the TextCraft benchmark (sea/env/benchmarks/textcraft.py).
```

### Example instantiation: Implement ExpeL (Experience + Learning)

```
I want to implement a new evolution method called ExpeL on the SEA platform.

**What it does**: Extracts generalizable "rules" from successful and failed
trajectories, stores them as structured insights (not raw reflections).

**What it evolves**: Memory (rules stored as high-quality semantic entries)

**How it works**:
1. Collect trajectories on the environment
2. For successful trajectories: extract "what worked and why" as a rule
3. For failed trajectories: extract "what to avoid and why" as a rule
4. Rules are more structured than reflections: "IF [condition] THEN [action] BECAUSE [reason]"
5. Store rules in memory with high priority scores
6. Agent retrieves relevant rules before each action

**Reference paper**: "ExpeL: LLM Agents Are Experiential Learners" (arXiv:2308.10144)

Please implement this as a new Evolver class in sea/evolution/methods/expel.py...
```

---

## Template 2: Implement a New Environment

```
I want to add a new environment called [ENV_NAME] to the SEA platform.

**What it is**: [DESCRIPTION OF THE ENVIRONMENT/BENCHMARK]

**Python package**: [PACKAGE NAME AND INSTALL COMMAND]

**API details**:
- Import: [IMPORT STATEMENT]
- Create env: [HOW TO CREATE]
- reset() returns: [RETURN TYPE]
- step() takes: [ACTION FORMAT]
- step() returns: [RETURN TYPE]
- Reward: [REWARD DESCRIPTION]
- Task types: [LIST OF TASK CATEGORIES, IF ANY]

Please implement the adapter in sea/env/benchmarks/[NAME].py:
- Inherit from `sea.env.base.SEAEnv`
- Register with `@ENV_REGISTRY.register("[NAME]")`
- Implement reset(), step(), get_task_ids(), name property
- Extract task_type in info dict if the env has task categories
- Handle the upstream package's specific return format

Key files to reference:
- sea/env/base.py — SEAEnv ABC
- sea/env/benchmarks/textcraft.py — TextCraft adapter (verified working)
- sea/env/benchmarks/alfworld.py — ALFWorld adapter
- sea/core/types.py — Observation, Action types
```

---

## Template 3: Implement a New Evolution Target

```
I want to make [COMPONENT] an evolution target on the SEA platform.

**What component**: [DESCRIPTION]

**State type**: [WHAT TYPE T SHOULD Evolvable[T] USE?]

**How it evolves**:
- get_evolvable_state() returns: [WHAT STATE IS EXPOSED TO EVOLVERS]
- set_evolvable_state() applies: [HOW NEW STATE IS APPLIED]

Please implement this by:
1. Create a new class implementing `Evolvable[T]` from `sea.core.base`
2. Implement get_evolvable_state(), set_evolvable_state(), evolution_metadata()
3. Also implement Checkpointable: save_checkpoint(), load_checkpoint(), state_dict()

Key files to reference:
- sea/core/base.py — Evolvable[T] protocol
- sea/evolution/targets/lm_params.py — LoRATarget example
- sea/evolution/targets/prompt.py — PromptTarget example
- sea/agent/memory/episodic.py — EpisodicMemory (Evolvable[list[dict]])
```

---

## Template 4: Implement Skill Composition

```
I want to implement [SKILL_COMPOSITION_METHOD] for agent skill evolution on SEA.

**Atomic skills**: [LIST OF BASE SKILLS]
**Composition logic**: [HOW SKILLS ARE COMBINED]
**When to compose**: [TRIGGER CONDITION]

The SEA platform already supports:
- TextSkill and ComposedSkill classes (sea/agent/skills/code_skill.py)
- SkillLibrary with FAISS retrieval and threshold (sea/agent/skills/library.py)
- SkillInfo has sub_skills and composition_plan fields (sea/agent/skills/base.py)

Please implement a custom SkillCompositionEvolver that:
1. Identifies frequently co-occurring atomic skills in trajectories
2. Uses a strong LLM to compose them into higher-level ComposedSkill objects
3. Stores them in the SkillLibrary
4. Handles deduplication with existing skills

Key files to reference:
- sea/agent/skills/code_skill.py — TextSkill, ComposedSkill classes
- sea/agent/skills/library.py — SkillLibrary (Evolvable)
- sea/evolution/base.py — Evolver ABC
```

---

## Template 5: Run an Experiment

```
I want to run a [METHOD] evolution experiment on the SEA platform.

**Setup**:
- Model: [MODEL NAME, e.g., Qwen/Qwen3.5-9B]
- Environment: [ENV, e.g., TextCraft / ALFWorld]
- Evolution method: [sft / rl / icl / prompt / custom]
- Evolution target: [lora / prompt / memory / skills]
- Backend: [vllm (local GPU) / api (OpenAI-compatible)]

**Experiment parameters**:
- Iterations: [N]
- Trajectories per iteration: [N]
- Eval episodes: [N]

Please create a runnable script in examples/[NAME]/run.py that:
1. Sets up the agent with the specified backend
2. Creates the environment
3. Runs the evolution loop: collect → evolve → evaluate
4. Logs results and saves summary.json

Use existing components:
- sea/agent/agent.py — SEAAgent
- sea/evolution/pipeline.py — EvolutionPipeline (or manual loop)
- sea/metrics/evaluator.py — Evaluator
- sea/evolution/data/trajectory.py — TrajectoryCollector

For local GPU experiments, use CUDA_VISIBLE_DEVICES to separate
inference (GPU 0) and training (GPU 1).
```

---

## Template 6: Implement Continual Learning

```
I want to implement continual learning for agents on the SEA platform.

**Method**: [e.g., O-LoRA, EWC, PackNet, Progressive Neural Networks]
**Task sequence**: [e.g., ALFWorld task types: pick → clean → heat → cool → examine]
**What to prevent**: Catastrophic forgetting of earlier tasks

The SEA platform provides:
- LoRATarget with multi-adapter tracking and r_sum (sea/evolution/targets/lm_params.py)
- Task-type aware trajectory collection (sea/evolution/data/trajectory.py)
- SFTEvolver/RLEvolver with trainer_callbacks and model_init_fn support
- SEAEnv.get_task_types() for environments with task categories

Please implement:
1. A ContinualLearningPipeline that trains on tasks sequentially
2. The specific continual learning method's constraint/regularizer
3. Evaluation after each task: test on ALL seen tasks (measure forgetting)
4. Comparison baseline: naive sequential LoRA (no forgetting prevention)

Key files to reference:
- sea/evolution/targets/lm_params.py — LoRATarget (adapter_history, r_sum)
- sea/evolution/methods/sft.py — SFTEvolver (trainer_callbacks, model_init_fn)
- sea/evolution/pipeline.py — EvolutionPipeline
```
