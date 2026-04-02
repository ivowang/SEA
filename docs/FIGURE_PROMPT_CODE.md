# Prompt for Generating SEA Code Architecture Figure

> For co-developers: a detailed code-level architecture diagram showing modules, classes, protocols, and data flow.

---

## Prompt

```
Create a detailed software architecture diagram for "SEA: Self-Evolving Agent Platform" aimed at co-developers. This should look like a professional engineering architecture diagram — showing Python packages, key classes, protocol inheritance, and data flow between modules. Think of it as an enhanced UML-style diagram with a modern flat aesthetic.

=== VISUAL STYLE ===
- White background, flat design, thin lines
- Color-coded by Python package:
  - sea/core: dark slate gray (#4A5568)
  - sea/agent: steel blue (#4682B4)
  - sea/env: forest green (#2E8B57)
  - sea/evolution: burnt orange (#CC6633)
  - sea/llm: dark purple (#6B46C1)
  - sea/metrics: teal (#2C7A7B)
- Each package is a large rounded rectangle with its color as a thin left-side accent bar (4px), white fill
- Classes inside packages shown as smaller rectangles with the class name in bold and key methods listed below in monospace font
- Inheritance/protocol lines: dashed with hollow triangle arrowhead
- Data flow lines: solid with filled arrowhead
- Composition lines: solid with diamond
- Font: monospace (Fira Code or Source Code Pro style), 8pt for method names, 10pt for class names, 12pt for package names
- Landscape orientation, approximately 10 inches wide × 7 inches tall (poster/wiki style, not paper-constrained)

=== LAYOUT: 6 PACKAGE ZONES ===

Arrange the 6 packages in a 3×2 grid:

```
┌──────────────────┬──────────────────┬──────────────────┐
│   sea/core        │   sea/agent       │   sea/env         │
│   (protocols)     │   (agent logic)   │   (environments)  │
├──────────────────┼──────────────────┼──────────────────┤
│   sea/evolution   │   sea/llm         │   sea/metrics      │
│   (training)      │   (LLM backends)  │   (evaluation)     │
└──────────────────┴──────────────────┴──────────────────┘
```

=== ZONE 1: sea/core (top-left, dark slate) ===

Title: "sea.core — Foundation Protocols & Types"

Show these classes/dataclasses:

1. «ABC» Checkpointable
   - save_checkpoint(path)
   - load_checkpoint(path)
   - state_dict() → dict

2. «ABC, Generic[T]» Evolvable[T] ——inherits from——▷ Checkpointable
   - get_evolvable_state() → T
   - set_evolvable_state(T)
   - evolution_metadata() → dict

3. «dataclass» Observation
   - text: str
   - available_actions: list[str]

4. «dataclass» Action
   - text: str
   - action_type: str
   - metadata: dict

5. «dataclass» Step
   - observation → action → next_observation
   - reward: float, done: bool

6. «dataclass» Trajectory
   - steps: list[Step]
   - task_id, task_type, total_reward, success

7. Registry
   - register(name) → decorator
   - build(name, **kwargs) → instance
   - Show small labels: ENV_REGISTRY, EVOLVER_REGISTRY, LLM_BACKEND_REGISTRY, ...

Draw a dashed inheritance arrow from Evolvable to Checkpointable.

=== ZONE 2: sea/agent (top-center, steel blue) ===

Title: "sea.agent — Agent Components"

This is the largest and most detailed zone. Show these classes with composition arrows:

1. SEAAgent (main class, bold border)
   - act(observation, task_desc, step) → Action
   - run_episode(env, task_id) → Trajectory
   - evolvable_components() → dict[str, Evolvable]
   - save/load_checkpoint(path)
   Composition arrows (diamond) going to:
   → LLMBrain
   → WorkingMemory
   → ReActPlanner
   → SkillLibrary
   → ToolRegistry

2. LLMBrain «Evolvable[dict]»
   - backend: LLMBackend
   - system_prompt: str
   - lora_name / lora_path
   - generate(messages) → GenerationOutput
   - swap_lora(path)
   Show a dashed line to sea/llm zone labeled "delegates to LLMBackend"

3. ReActPlanner
   - SYSTEM_PROMPT (constant)
   - _history: list[dict]
   - plan(brain, context) → Action
   - _parse_action(text) → Action
   - reset()
   Show: "Thought → Action" parsing flow as a tiny inline note

4. «Evolvable[list[dict]]» WorkingMemory
   - _buffer: deque[MemoryEntry]
   - add(entry), retrieve(query, k) → list
   - get/set_evolvable_state()
   Show small badge: "Evolvable"

5. «Evolvable[list[dict]]» SkillLibrary
   - _skills: dict[str, SkillMd]
   - _skills_dir: Path (file-system)
   - add_skill(Skill | SkillMd | dict)
   - retrieve(query, k, level) → list[SkillView]
   - get_index() → list[SkillView] (INDEX level)
   - retrieve_full(name) → SkillView (FULL level)
   - _keyword_retrieve() / _embedding_retrieve()
   Show small badge: "Evolvable"
   Show a tiny sub-box: "SKILL.md Format" with "--- YAML frontmatter --- \n ## Steps \n 1. ..."

6. SkillMd (dataclass)
   - frontmatter: SkillFrontmatter (name, description, tags, when_to_use)
   - body: str (markdown)
   Connected to SkillLibrary

7. DisclosureLevel (enum)
   - INDEX | SUMMARY | FULL
   Connected to SkillView

8. ToolRegistry
   - _tools: dict[str, Tool]
   - register(tool), execute(tool_name, **kwargs) → ToolResult
   Show inside: "CalculatorTool", "ReadSkillTool"

9. «dataclass» PlanningContext
   - observation, retrieved_memories, retrieved_skills
   - skill_index, available_tools
   - task_description, step_number
   Arrow from PlanningContext into ReActPlanner

Draw the internal data flow:
- SEAAgent.act() calls: Memory.retrieve() → SkillLibrary.get_index() + retrieve() → build PlanningContext → Planner.plan() → if tool_call: ToolRegistry.execute() → re-plan
- Show this as a numbered flow (①②③④⑤) with thin arrows

=== ZONE 3: sea/env (top-right, forest green) ===

Title: "sea.env — Environment Adapters"

1. «ABC» SEAEnv
   - reset(seed, task_id) → (Observation, info)
   - step(action) → (Observation, reward, terminated, truncated, info)
   - get_task_ids() → list[str]
   - get_task_types() → list[str]
   - close()

2. TextCraftEnv «SEAEnv»
   - Minecraft crafting recipes
   - task_id: seed_0..seed_N

3. ALFWorldEnv «SEAEnv»
   - 6 task types: pick, clean, heat, cool, examine, pick_two
   - _game_index: dict[str, str] (task_id → game_file)
   - Per-game selection via gamefiles swap
   - get_task_ids() returns real selectable IDs

4. WebShopEnv «SEAEnv»
   - Web shopping sessions
   - task_id = session ID

5. GymnasiumWrapper «SEAEnv»
   - Wraps any gym.Env

6. FunctionEnv «SEAEnv»
   - Defined by reset_fn / step_fn callables

Draw inheritance arrows from all concrete envs to SEAEnv ABC.

=== ZONE 4: sea/evolution (bottom-left, burnt orange) ===

Title: "sea.evolution — Evolution Methods, Data & Pipeline"

This zone has 3 sub-sections:

**Sub-section A: Methods**

1. «ABC» Evolver «Checkpointable»
   - evolve(agent, target: Evolvable, trajectories, metrics)
   - requires_trajectories() → bool

2. SFTEvolver «Evolver»
   - _tokenize_chat_data() — manual assistant-only label masking
   - Uses Trainer (not SFTTrainer) for correct Qwen chat template handling
   - Target: Evolvable[Path] (LoRATarget)

3. RLEvolver «Evolver»
   - algorithm: "reinforce" | "dpo"
   - _evolve_reinforce(): custom compute_loss = -log_prob × advantage
   - _evolve_dpo(): TRL DPOTrainer
   - _tokenize_reinforce_data(): separate prompt/completion tokenization
   - Target: Evolvable[Path] (LoRATarget)

4. ICLEvolver «Evolver»
   - _generate_reflection(agent, trajectory)
   - _select_exemplars(successful_trajectories)
   - _extract_skills_from_trajectories() (optional)
   - Target: Evolvable[list[dict]] (Memory)

5. ExpeLEvolver «Evolver»
   - Extracts IF → THEN → BECAUSE rules
   - Target: Evolvable[list[dict]] (Memory)

6. PromptEvolver «Evolver»
   - Generates N prompt variants, evaluates, selects best
   - Target: Evolvable[str] (PromptTarget)

Draw inheritance arrows from all concrete evolvers to Evolver ABC.

**Sub-section B: Data Pipeline**

7. TrajectoryCollector
   - collect(agent, envs, n) — serial, multi-env (env,task_id) pairing
   - collect_parallel(agent_factory, env_factory, n, max_workers) — ThreadPool, real-time target tracking
   - collect_subprocess(...) — process isolation, JSONL atomic writes

8. TrajectoryBuffer
   - _buffer: deque[Trajectory]
   - sample(n, filter_fn), successful(), by_task_type()

9. Data Conversion Functions (show as a box with function names):
   - trajectories_to_sft_data() → list[{"messages": [...]}]
   - trajectories_to_reinforce_data() → list[{context, action, advantage}]
   - trajectories_to_preference_pairs() → list[{prompt, chosen, rejected}]
   - compute_returns(rewards, gamma) → list[G_t]

Draw arrows: TrajectoryCollector → TrajectoryBuffer → Data Conversion → Evolvers

**Sub-section C: Evolution Targets**

10. LoRATarget «Evolvable[Path]»
    - adapter_dir: Path
    - adapter_history: list
    - get_evolvable_state() → Path (adapter checkpoint)

11. PromptTarget «Evolvable[str]»
    - prompt_text: str
    - history: list[(prompt, score)]

12. MemoryTarget «Evolvable[list[dict]]»
    - Delegates to wrapped Memory

13. SkillTarget «Evolvable[list[dict]]»
    - Delegates to wrapped SkillLibrary

Draw dashed arrows from each Target back up to the corresponding Agent component.

**Pipeline:**

14. EvolutionPipeline
    - run(): for each iteration: collect → evolve → evaluate → checkpoint
    - agent, envs, evolvers, evaluator, metrics, config
    - extra_targets: dict (for LoRATarget, PromptTarget)

=== ZONE 5: sea/llm (bottom-center, dark purple) ===

Title: "sea.llm — LLM Backends"

1. «ABC» LLMBackend
   - model_name: str
   - generate(messages, temperature, max_tokens, lora_name) → GenerationOutput
   - generate_batch(batches) → list[GenerationOutput]
   - supports_lora() → bool
   - load_lora(path, name) / unload_lora(name)

2. APIBackend «LLMBackend»
   - OpenAI-compatible client
   - timeout, max_retries (built-in retry)
   - No LoRA support

3. VLLMBackend «LLMBackend»
   - vLLM engine with PagedAttention
   - tensor_parallel_size
   - LoRA hot-swap: _active_loras dict
   - Raises error if requested LoRA not loaded

4. HFTrainingBackend (not LLMBackend — training only)
   - get_trainable_model(adapter_path, lora_config) → PeftModel
   - get_tokenizer() → AutoTokenizer
   - save_adapter(model, path)
   - Supports 4-bit / 8-bit quantization
   - enable_input_require_grads() for gradient checkpointing

Draw: LLMBrain (in agent zone) ——delegates to——→ LLMBackend (in llm zone)
Draw: SFTEvolver / RLEvolver (in evolution zone) ——uses——→ HFTrainingBackend

=== ZONE 6: sea/metrics (bottom-right, teal) ===

Title: "sea.metrics — Evaluation & Reporting"

1. Evaluator
   - evaluate(agent, envs) → EvalResults
   - num_episodes_per_env, eval_temperature
   - Aggregates: success_rate, avg_reward, per_env, per_task

2. «dataclass» EvalResults
   - success_rate, avg_reward, avg_steps, num_episodes
   - per_env: dict, per_task: dict
   - trajectories: list[Trajectory]

3. MetricsTracker
   - reporters: list[Reporter]
   - log(dict), log_eval(EvalResults)
   - global_step: int

4. Reporters (show as 3 small boxes):
   - ConsoleReporter (rich table)
   - TensorBoardReporter
   - WandBReporter

=== CROSS-ZONE ARROWS (important!) ===

Draw these prominent cross-zone data flow arrows:

1. sea/agent → sea/env: "agent.run_episode(env)" (bidirectional thick arrow)
2. sea/env → sea/evolution/data: "Trajectory" (solid arrow)
3. sea/evolution/data → sea/evolution/methods: "training data" (solid arrow)
4. sea/evolution/methods → sea/evolution/targets: "update state" (solid arrow)
5. sea/evolution/targets → sea/agent: "improve agent" (dashed arrow, going upward)
6. sea/llm → sea/agent: "LLMBackend" (composition arrow)
7. sea/llm → sea/evolution: "HFTrainingBackend" (usage arrow)
8. sea/metrics → sea/agent + sea/env: "Evaluator.evaluate()" (thin arrow)

=== LEGEND (bottom-right corner) ===

Small legend box:
- Solid arrow with diamond: "Composition (has-a)"
- Dashed arrow with hollow triangle: "Inheritance (is-a)"
- Solid arrow with filled head: "Data flow"
- Dashed arrow with filled head: "Feedback / update"
- Small colored squares showing the 6 package colors

=== TITLE ===

Top of figure:
"SEA Platform — Code Architecture"
Subtitle: "Python Package Structure, Key Classes, and Data Flow"

=== IMPORTANT NOTES ===
- This is a DEVELOPER-facing diagram, so show real class names, method signatures, and type annotations
- Every Evolvable component should have a small "Evolvable[T]" badge showing its type parameter
- The Protocol hierarchy (Checkpointable → Evolvable) should be visually prominent in the core zone
- Make sure the cross-zone arrows are clearly visible — they tell the story of how data flows through the system
- Keep method lists to 3-5 most important methods per class, don't list everything
- Monospace font for all code elements (class names, method names, type parameters)
```

---

## Simplified Version

If the full diagram is too dense, generate a simplified version with this instruction appended:

```
SIMPLIFICATION: Instead of showing all methods, show only class names with 1-line descriptions. Remove the internal agent data flow numbering. Merge the Data Pipeline sub-section into a single "TrajectoryCollector + DataConversion" box. This should fit in a single slide or wiki page.
```
