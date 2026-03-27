# Tutorial: RL (GRPO) Evolution on TextCraft

Train a LoRA adapter via GRPO with **environment-backed rewards**: the model generates crafting actions, which are parsed and executed in TextCraft to compute real rewards. This is the key distinction from standard GRPO — the reward signal comes from actual task success, not heuristics.

**Requires**: 2 GPUs (inference + training), Qwen3.5-9B model

## Quick Run

```bash
export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
export SEA_INFERENCE_GPU="4"
export SEA_TRAINING_GPU="5"
python examples/rl_textcraft/run.py
```

## How It Works

```
For each iteration:
  1. Collect trajectories to build prompt dataset (task descriptions)
  2. GRPOTrainer generates N completions per prompt
  3. Environment-backed reward function:
     a. Parse completion → extract "Action: ..." lines
     b. Reset TextCraft with the task's seed
     c. Execute parsed actions in the environment
     d. Return actual cumulative reward (0.0 or 1.0)
  4. GRPO uses group-relative rewards to update LoRA
  5. Hot-swap and evaluate
```

## The GRPO Environment Reward

Standard TRL GRPOTrainer calls `reward_fn(completions: list[str])` on raw text. SEA bridges this to real environment execution:

```python
# sea/evolution/methods/rl.py — RLEvolver._make_env_reward_fn()

def reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        # 1. Parse actions from ReAct format
        actions = parse_actions_from_completion(completion)
        # 2. Reset env with the task
        task_id = prompt_to_task_id.get(prompts[i]) if prompts else None
        obs, info = env.reset(task_id=task_id)
        # 3. Execute actions
        total_reward = 0.0
        for action_text in actions:
            obs, reward, done, trunc, info = env.step(Action(text=action_text))
            total_reward += reward
            if done or trunc:
                break
        rewards.append(total_reward)
    return rewards
```

The action parser handles ReAct-format output:
```
Thought: I need oak logs to make planks
Action: get 1 oak log           ← extracted
Thought: Now craft planks
Action: craft 4 oak planks using 1 oak log  ← extracted
```

## Key Code

### Setup with environment-backed reward

```python
from sea.evolution.methods.rl import RLEvolver

reward_env = TextCraftEnv(max_steps_val=15)

evolver = RLEvolver(
    model_name=MODEL_PATH,
    algorithm="grpo",
    device="cuda:1",
    num_generations=4,        # generate 4 completions per prompt
    max_completion_length=512,
    envs=[reward_env],        # environment for reward computation
)
```

### Evolution loop

```python
for iteration in range(NUM_ITERATIONS):
    trajectories = collector.collect(agent, [env], n=20)
    evolver.evolve(agent, lora_target, trajectories, metrics, envs=[reward_env])
    # New adapter automatically hot-swapped
    result = evaluator.evaluate(agent, [env])
```

## GRPO vs DPO

| | GRPO | DPO |
|---|---|---|
| **Reward source** | Environment execution (online) | Preference pairs from trajectories (offline) |
| **Data** | Prompts only | (prompt, chosen, rejected) triples |
| **When to use** | When env is available and fast | When env is slow or unavailable |

Both are available via `RLEvolver(algorithm="grpo")` or `RLEvolver(algorithm="dpo")`.

## Full Script

See `examples/rl_textcraft/run.py`
