# Tutorial: SFT Evolution on TextCraft

Train a LoRA adapter via supervised fine-tuning on successful TextCraft crafting trajectories. The agent collects crafting experiences, filters successes, trains LoRA weights, and hot-swaps the improved adapter — all on the real TextCraft benchmark.

**Requires**: 2 GPUs (inference + training), Qwen3.5-9B model

## Quick Run

```bash
export SEA_MODEL_PATH="/root/models/Qwen3.5-9B"
export SEA_INFERENCE_GPU="4"
export SEA_TRAINING_GPU="5"
python examples/sft_textcraft/run.py
```

## How It Works

```
For each iteration:
  1. Agent interacts with TextCraft via vLLM (GPU 0)
     → Tries "get 1 oak log", "craft 4 oak planks using 1 oak log", etc.
  2. Successful trajectories filtered (reward > 0)
  3. Convert to multi-turn chat format for SFT
  4. Train LoRA adapter on GPU 1 (PEFT + TRL SFTTrainer)
  5. Hot-swap new adapter into vLLM (zero downtime)
  6. Evaluate improvement
```

## Key Code

### Collect and filter

```python
trajectories = collector.collect(agent, [env], n=20)
good = [t for t in trajectories if t.success or t.total_reward > 0]
sft_data = trajectories_to_sft_data(good, system_prompt=agent.brain.system_prompt)
dataset = to_hf_dataset(sft_data)
```

### Train LoRA

```python
hf = HFTrainingBackend(model_name=MODEL_PATH, device="cuda:1", torch_dtype="bfloat16")
model = hf.get_trainable_model(lora_config={
    "r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
})

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(num_train_epochs=2, learning_rate=2e-5, max_length=2048, bf16=True, ...),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Hot-swap

```python
hf.save_adapter(model, adapter_path)
agent.brain.backend.load_lora(str(adapter_path), name="iter_1")
agent.brain.lora_name = "iter_1"
```

## TextCraft Environment

TextCraft is a text-based Minecraft crafting benchmark (`pip install textcraft`). The agent receives a goal item and available recipes, then must execute `get` and `craft` commands in the correct order.

```
Observation:
  Crafting commands:
  craft 1 dark oak sign using 6 dark oak planks, 1 stick
  craft 4 dark oak planks using 1 dark oak log
  craft 4 sticks using 2 dark oak planks

  Goal: craft dark oak sign.

Actions: "get 1 dark oak log", "craft 4 dark oak planks using 1 dark oak log", ...
Reward: 1.0 on crafting the goal item, 0.0 otherwise
```

## Full Script

See `examples/sft_textcraft/run.py`
