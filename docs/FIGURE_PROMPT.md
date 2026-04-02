# Prompt for Generating SEA Architecture Figure

> Use this prompt with Gemini (or similar AI image generation) to create a publication-quality framework diagram for the SEA platform.

---

## Prompt

```
Create a publication-quality academic framework diagram for "SEA: Self-Evolving Agent Platform". This is a research platform where an LLM-based agent interacts with environments, collects experience trajectories, and evolves its own components (LoRA weights, memory, skills, prompts) through various training methods. The figure should be detailed enough to serve as the main architecture figure in a top-venue ML paper (NeurIPS/ICML style).

=== VISUAL STYLE ===
- White background, flat design, no 3D effects, no heavy gradients
- Soft pastel fills: light blue for Agent, light green for Environment, light orange for Evolution, light purple for Targets, light gray for Infrastructure
- Thin black or dark gray outlines (1-1.5pt)
- Rounded rectangles for all blocks
- Clean sans-serif font (like Helvetica or Arial), 8-10pt for labels
- Arrows: solid black with arrowheads for data flow, dashed for "feedback/update" flows
- The figure should be landscape orientation, roughly 7 inches wide × 4.5 inches tall (two-column paper width)

=== OVERALL LAYOUT (3 horizontal bands) ===

**Top band: Agent ↔ Environment interaction**
**Middle band: Trajectory Collection → Data Processing → Evolution Methods**
**Bottom band: Evolution Targets + Infrastructure**

=== DETAILED COMPONENT LAYOUT ===

--- TOP-LEFT: AGENT BLOCK (light blue, large) ---

A large rounded rectangle labeled "SEAAgent" at the top. Inside it, arrange these sub-components as nested rounded rectangles:

1. "LLM Brain" (center-top, slightly larger than others)
   - Inside: a small label "LLM" with a gear icon
   - Below it: "System Prompt" (small italic text)
   - Below it: "LoRA Adapter" (small italic text with a small plug icon)

2. "ReAct Planner" (to the right of Brain)
   - Small label inside: "Thought → Action"
   - An arrow from Brain to Planner labeled "generate"
   - An arrow from Planner back to Brain labeled "re-plan"

3. "Memory" (below-left of Brain)
   - Inside: "WorkingMemory" as primary
   - Small text: "retrieve(query) → relevant entries"
   - A small "Evolvable" badge/tag on the corner

4. "Skill Library" (below-center of Brain)
   - Inside: show 3 tiny stacked document icons representing SKILL.md files
   - Labels for the 3 progressive disclosure levels in a tiny vertical list:
     - "INDEX: name + description"
     - "SUMMARY: + steps outline"
     - "FULL: complete .md content"
   - A small "Evolvable" badge/tag on the corner
   - A tiny "ReadSkillTool" label with an arrow pointing from Planner to Skill Library

5. "Tool Registry" (below-right of Brain)
   - Inside: tiny labels "Calculator", "ReadSkill", "Custom..."

Show internal arrows:
- Memory → Planner (labeled "context")
- Skill Library → Planner (labeled "skill index + summaries")
- Tool Registry → Planner (labeled "tool results")

--- TOP-RIGHT: ENVIRONMENT BLOCK (light green) ---

A rounded rectangle labeled "Environment (SEAEnv)". Inside, show:
- "reset(task_id)" and "step(action)" as two API labels
- Below: three small environment cards arranged horizontally:
  - "TextCraft" with a tiny pickaxe icon (Minecraft crafting)
  - "ALFWorld" with a tiny house icon (household tasks)
  - "WebShop" with a tiny cart icon (web shopping)
- A small label below: "Observation, Reward, Done"

Show arrows between Agent and Environment:
- Agent → Environment: thick arrow labeled "Action" (going right)
- Environment → Agent: thick arrow labeled "Observation" (going left)
- These two arrows should form a clear bidirectional interaction loop

--- MIDDLE BAND: DATA PIPELINE (light gray background strip) ---

Spanning the full width, a horizontal pipeline with these stages connected by arrows:

1. "Trajectory Collection" (left)
   - Inside, show 3 small mode labels stacked:
     - "collect() — serial"
     - "collect_parallel() — API concurrent"
     - "collect_subprocess() — process isolation"
   - A downward arrow from the Agent-Environment loop above, labeled "episodes"

2. "Trajectory Buffer" (center-left)
   - A small cylinder/database icon
   - Label: "filter: success/reward threshold"

3. "Data Conversion" (center)
   - Three output branches shown as small arrows fanning out:
     - "→ SFT data (chat messages)"
     - "→ REINFORCE data (context, action, advantage)"
     - "→ DPO pairs (chosen vs rejected)"

4. "Reward Computation" (center-right, small)
   - Labels: "Environment reward", "Success binary", "LLM Judge"

Arrow flow: Trajectory Collection → Buffer → Data Conversion → (feeds into Evolution Methods below)

--- BOTTOM-LEFT: EVOLUTION METHODS BLOCK (light orange, wide) ---

A wide rounded rectangle labeled "Evolution Methods". Inside, show 6 method cards arranged in a 2×3 grid:

Row 1 (Parameter-updating methods):
- "SFT" — subtitle: "LoRA fine-tuning on successful trajectories, assistant-only loss masking"
- "REINFORCE" — subtitle: "Offline policy gradient, loss = -log π(a|s) × advantage"
- "DPO" — subtitle: "Direct preference optimization from trajectory pairs"

Row 2 (Non-parametric methods):
- "ICL / Reflexion" — subtitle: "Verbal reflections on failures + few-shot exemplars from successes"
- "ExpeL" — subtitle: "Extract IF→THEN→BECAUSE semantic rules"
- "Prompt Optimization" — subtitle: "LLM-based prompt mutation and selection"

Show an arrow from Data Conversion (above) feeding into this block.

--- BOTTOM-RIGHT: EVOLUTION TARGETS (light purple) ---

A rounded rectangle labeled "Evolution Targets (Evolvable[T])". Inside, show 4 target cards vertically:

1. "LoRA Weights" — "T = Path" — arrow going up to Agent's "LoRA Adapter" (dashed, labeled "hot-swap")
2. "System Prompt" — "T = str" — arrow going up to Agent's "System Prompt" (dashed)
3. "Memory Entries" — "T = list[dict]" — arrow going up to Agent's "Memory" (dashed)
4. "SKILL.md Files" — "T = list[dict]" — arrow going up to Agent's "Skill Library" (dashed)

Show arrows from Evolution Methods → Evolution Targets (labeled "update")
Show dashed arrows from Evolution Targets → Agent sub-components (labeled "improve")

--- BOTTOM STRIP: INFRASTRUCTURE (very light gray, thin) ---

A thin horizontal strip at the very bottom showing supporting infrastructure:

Left section - "LLM Backends":
- Three small boxes: "API Backend (remote)" | "vLLM (local inference)" | "HF Training (PEFT LoRA)"
- An arrow going up to Agent's LLM Brain

Center section - "Metrics & Evaluation":
- "Evaluator" | "MetricsTracker"
- Small labels: "Console, TensorBoard, W&B"

Right section - "Configuration":
- "YAML + OmegaConf" | "Registry Pattern"

=== KEY ANNOTATIONS ===

Add these text annotations near the relevant components (in a slightly smaller, italic font):

1. Near the Agent-Environment loop: "Episode: obs → act → obs → act → ... → done"
2. Near collect_parallel(): "30+ concurrent API workers"
3. Near the SKILL.md in Skill Library: "Progressive Disclosure (2025)"
4. Near LoRA Weights target: "Evolvable[Path]"
5. Near the overall diagram: a small circled number flow showing the evolution loop order:
   ① Agent acts in Environment
   ② Trajectories collected
   ③ Data converted for training
   ④ Evolver updates target
   ⑤ Improved agent acts again

=== TITLE ===

At the very top of the figure, centered:
"SEA: Self-Evolving Agent Platform"
Subtitle (smaller): "A Modular Research Platform for Agent Self-Improvement"

=== IMPORTANT NOTES ===
- Make sure ALL text is legible at the final size (minimum 7pt font)
- The evolution loop (①→②→③→④→⑤→①) should be visually prominent — this is the core contribution
- The Agent block should be the largest and most detailed component since it is the most complex
- Use consistent color coding throughout: blue=agent, green=env, orange=evolution, purple=targets, gray=infra
- Keep the layout clean and uncluttered — use whitespace effectively
- All arrows should have clear directionality and labels
```

---

## Usage Notes

- **Gemini**: Paste the prompt above directly. May need 2-3 iterations to get the layout right.
- **If the layout is too crowded**: Ask Gemini to "make the figure wider" or "reduce the font size to 7pt and add more whitespace".
- **If text is illegible**: Ask to "enlarge all text labels to at least 9pt".
- **If you want a simpler version**: Remove the Infrastructure bottom strip and the Data Pipeline middle band, keeping only the Agent ↔ Environment ↔ Evolution triangle.
- **For a paper submission**: After generating, manually refine in Illustrator/Figma/draw.io for perfect alignment.
