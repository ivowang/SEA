"""RL Evolver: reinforcement learning with REINFORCE/DPO.

Uses pre-collected trajectories for offline policy gradient (REINFORCE)
or Direct Preference Optimization (DPO).

REINFORCE: computes per-step log P(action | context) * advantage from
collected trajectories. No online generation needed — rewards come from
the trajectories themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.evolution.data.dataset import (
    to_hf_dataset,
    trajectories_to_preference_pairs,
    trajectories_to_reinforce_data,
)

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: parse actions from ReAct-format completions
# ---------------------------------------------------------------------------

def parse_actions_from_completion(completion: str) -> list[str]:
    """Parse action strings from a ReAct-format LLM completion.

    Handles formats like:
        Thought: I need oak logs
        Action: get 1 oak log

    Returns a list of action strings (without the "Action:" prefix).
    If no "Action:" lines found, treats the entire text as a single action.
    """
    import re

    actions = []
    for match in re.finditer(r"Action:\s*(.+?)(?:\n|$)", completion):
        action_text = match.group(1).strip()
        if action_text and not action_text.startswith("finish"):
            actions.append(action_text)
    if not actions:
        clean = completion.strip()
        if clean:
            actions.append(clean)
    return actions


@EVOLVER_REGISTRY.register("rl")
class RLEvolver(Evolver):
    """RL-based evolution using REINFORCE or DPO.

    REINFORCE: offline trajectory-level policy gradient. Uses pre-collected
    trajectories with real environment rewards. Each step's advantage is
    G_t (discounted return) normalized across the batch.

    DPO: offline preference optimization from trajectory pairs.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        algorithm: str = "reinforce",  # "reinforce", "grpo" (alias), "dpo"
        device: str = "cuda:1",
        learning_rate: float = 1e-5,
        num_epochs: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 1024,
        kl_coeff: float = 0.1,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        output_dir: str = "outputs/rl",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        lora_config: dict[str, Any] | None = None,
        trainer_callbacks: list | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._algorithm = algorithm
        self._device = device
        self._lr = learning_rate
        self._epochs = num_epochs
        self._batch_size = batch_size
        self._grad_accum = gradient_accumulation_steps
        self._max_seq_length = max_seq_length
        self._kl_coeff = kl_coeff
        self._gamma = gamma
        self._entropy_coeff = entropy_coeff
        self._output_dir = Path(output_dir)
        self._torch_dtype = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._lora_config = lora_config
        self._callbacks = trainer_callbacks or []
        self._train_step = 0

    def requires_trajectories(self) -> bool:
        return True

    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        **kwargs: Any,
    ) -> None:
        if self._algorithm in ("reinforce", "grpo"):
            self._evolve_reinforce(agent, target, trajectories, metrics)
        elif self._algorithm == "dpo":
            self._evolve_dpo(agent, target, trajectories, metrics)
        else:
            raise ValueError(f"Unsupported algorithm: {self._algorithm}")

    # ------------------------------------------------------------------
    # REINFORCE: offline trajectory-level policy gradient
    # ------------------------------------------------------------------

    def _evolve_reinforce(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
    ) -> None:
        """Offline REINFORCE using pre-collected trajectories.

        Steps:
        1. Convert trajectories → (context, action, advantage) triples
        2. Tokenize context as prompt (labels=-100), action as completion
        3. Compute loss = -sum(log_prob(action_tokens)) * advantage
        4. Train with standard gradient descent + optional entropy bonus
        """
        from transformers import Trainer, TrainingArguments
        from sea.llm.hf_backend import HFTrainingBackend

        # 1. Convert trajectories to REINFORCE training data
        train_data = trajectories_to_reinforce_data(
            trajectories,
            system_prompt=agent.brain.system_prompt,
            gamma=self._gamma,
        )
        if not train_data:
            logger.warning("No training data for REINFORCE")
            return

        # 2. Load model with LoRA
        hf = HFTrainingBackend(
            model_name=self._model_name,
            device=self._device,
            torch_dtype=self._torch_dtype,
            load_in_4bit=self._load_in_4bit,
        )
        current_adapter = None
        try:
            state = target.get_evolvable_state()
            if isinstance(state, Path) and state.exists():
                current_adapter = state
        except Exception:
            pass
        model = hf.get_trainable_model(
            adapter_path=current_adapter,
            lora_config=self._lora_config,
        )
        tokenizer = hf.get_tokenizer()

        # 3. Tokenize into a dataset
        dataset = self._tokenize_reinforce_data(train_data, tokenizer)
        logger.info("REINFORCE dataset: %d samples", len(dataset))

        # 4. Custom trainer with REINFORCE loss
        entropy_coeff = self._entropy_coeff

        class _ReinforceTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]
                advantages = inputs["advantages"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Shift for causal LM (predict next token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Per-token log probs
                log_probs = F.log_softmax(shift_logits, dim=-1)
                # Gather log prob of actual tokens (clamp to avoid -100 index)
                token_log_probs = log_probs.gather(
                    -1, shift_labels.unsqueeze(-1).clamp(min=0)
                ).squeeze(-1)

                # Mask: only action tokens (where labels != -100)
                action_mask = (shift_labels != -100).float()

                # Per-sample sum of log probs over action tokens
                seq_log_probs = (token_log_probs * action_mask).sum(dim=-1)

                # REINFORCE loss: -log_prob * advantage
                loss = -(seq_log_probs * advantages).mean()

                # Optional entropy bonus for exploration
                if entropy_coeff > 0:
                    probs = log_probs.exp()
                    ent = -(probs * log_probs).sum(-1)
                    ent = (ent * action_mask).sum(-1) / action_mask.sum(-1).clamp(min=1)
                    loss = loss - entropy_coeff * ent.mean()

                return (loss, outputs) if return_outputs else loss

        self._train_step += 1
        run_dir = self._output_dir / f"reinforce_step_{self._train_step}"

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False,
            bf16=(self._torch_dtype == "bfloat16"),
            fp16=(self._torch_dtype == "float16"),
            dataloader_pin_memory=False,
        )

        trainer = _ReinforceTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=self._callbacks,
        )

        logger.info(
            "Starting REINFORCE training: %d samples, %d epochs",
            len(dataset), self._epochs,
        )
        trainer.train()

        # Save adapter and hot-swap
        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": "reinforce",
            "rl/num_samples": len(dataset),
            "rl/train_step": self._train_step,
            "rl/avg_advantage": sum(r["advantage"] for r in train_data) / len(train_data),
        })

        del model, trainer
        torch.cuda.empty_cache()

    def _tokenize_reinforce_data(
        self,
        data: list[dict[str, Any]],
        tokenizer,
    ) -> Any:
        """Tokenize REINFORCE data into a torch Dataset.

        For each record:
        - Tokenize context_messages as prompt (labels set to -100)
        - Tokenize action_text as completion (real labels)
        - Store advantage as a float tensor
        """
        from torch.utils.data import Dataset as TorchDataset

        # Apply chat template for context, then append action
        processed = []
        for record in data:
            # Build prompt from context messages
            context = record["context_messages"]
            action_text = record["action_text"]
            advantage = record["advantage"]

            # Tokenize context (prompt) — no generation, just input
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt_text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in context
                ) + "\nassistant: "

            # Tokenize prompt + action together
            full_text = prompt_text + action_text
            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=self._max_seq_length,
                return_tensors="pt",
            )
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self._max_seq_length,
                return_tensors="pt",
            )

            input_ids = full_tokens["input_ids"].squeeze(0)
            attention_mask = full_tokens["attention_mask"].squeeze(0)
            prompt_len = prompt_tokens["input_ids"].shape[1]

            # Labels: -100 for prompt tokens, real ids for action tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "advantages": torch.tensor(advantage, dtype=torch.float32),
            })

        class ReinforceDataset(TorchDataset):
            def __init__(self, items, pad_id):
                self.items = items
                self.pad_id = pad_id

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        # Pad to same length within dataset
        if processed:
            max_len = max(item["input_ids"].shape[0] for item in processed)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            for item in processed:
                cur_len = item["input_ids"].shape[0]
                if cur_len < max_len:
                    pad_len = max_len - cur_len
                    item["input_ids"] = F.pad(item["input_ids"], (0, pad_len), value=pad_id)
                    item["attention_mask"] = F.pad(item["attention_mask"], (0, pad_len), value=0)
                    item["labels"] = F.pad(item["labels"], (0, pad_len), value=-100)

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        return ReinforceDataset(processed, pad_id)

    # ------------------------------------------------------------------
    # DPO (offline, from preference pairs)
    # ------------------------------------------------------------------

    def _evolve_dpo(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
    ) -> None:
        """DPO: train on preference pairs from trajectories."""
        from trl import DPOConfig, DPOTrainer
        from sea.llm.hf_backend import HFTrainingBackend

        pairs = trajectories_to_preference_pairs(trajectories)
        if not pairs:
            logger.warning("No preference pairs for DPO")
            return

        dataset = to_hf_dataset(pairs)

        hf = HFTrainingBackend(
            model_name=self._model_name,
            device=self._device,
            torch_dtype=self._torch_dtype,
            load_in_4bit=self._load_in_4bit,
        )
        current_adapter = None
        try:
            state = target.get_evolvable_state()
            if isinstance(state, Path) and state.exists():
                current_adapter = state
        except Exception:
            pass
        model = hf.get_trainable_model(
            adapter_path=current_adapter,
            lora_config=self._lora_config,
        )
        tokenizer = hf.get_tokenizer()

        self._train_step += 1
        run_dir = self._output_dir / f"dpo_step_{self._train_step}"

        config = DPOConfig(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            beta=self._kl_coeff,
            logging_steps=10,
            save_strategy="no",
            bf16=(self._torch_dtype == "bfloat16"),
            fp16=(self._torch_dtype == "float16"),
        )

        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training: %d pairs", len(pairs))
        trainer.train()

        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": "dpo",
            "rl/num_pairs": len(pairs),
            "rl/train_step": self._train_step,
        })

        del model, trainer
        torch.cuda.empty_cache()
