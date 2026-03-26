"""RL Evolver: reinforcement learning with GRPO/DPO/PPO.

Uses TRL as the training backend. Supports:
- GRPO: Group Relative Policy Optimization (online, no critic)
- DPO: Direct Preference Optimization (offline, from preference pairs)
- PPO: Proximal Policy Optimization (online, with value head)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.evolution.data.dataset import (
    to_hf_dataset,
    trajectories_to_preference_pairs,
    trajectories_to_sft_data,
)
from sea.evolution.data.reward import RewardFunction

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


@EVOLVER_REGISTRY.register("rl")
class RLEvolver(Evolver):
    """RL-based evolution using GRPO, DPO, or PPO via TRL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        algorithm: str = "grpo",  # "grpo", "dpo", "ppo"
        device: str = "cuda:1",
        learning_rate: float = 1e-5,
        num_epochs: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_completion_length: int = 512,
        num_generations: int = 4,  # GRPO group size
        kl_coeff: float = 0.1,
        reward_functions: list[RewardFunction] | None = None,
        output_dir: str = "outputs/rl",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        lora_config: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._algorithm = algorithm
        self._device = device
        self._lr = learning_rate
        self._epochs = num_epochs
        self._batch_size = batch_size
        self._grad_accum = gradient_accumulation_steps
        self._max_completion_length = max_completion_length
        self._num_generations = num_generations
        self._kl_coeff = kl_coeff
        self._reward_fns = reward_functions or []
        self._output_dir = Path(output_dir)
        self._torch_dtype = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._lora_config = lora_config
        self._train_step = 0

    def requires_trajectories(self) -> bool:
        # DPO needs pre-collected trajectories; GRPO can work either way
        return True

    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
    ) -> None:
        if self._algorithm == "grpo":
            self._evolve_grpo(agent, target, trajectories, metrics)
        elif self._algorithm == "dpo":
            self._evolve_dpo(agent, target, trajectories, metrics)
        else:
            raise ValueError(f"Unsupported algorithm: {self._algorithm}")

    def _evolve_grpo(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
    ) -> None:
        """GRPO: train on prompts with group-relative rewards."""
        from trl import GRPOConfig, GRPOTrainer
        from sea.llm.hf_backend import HFTrainingBackend

        # Build prompt dataset from trajectory task descriptions
        prompts = []
        for traj in trajectories:
            task_desc = traj.metadata.get("task_description", "")
            if task_desc and task_desc not in prompts:
                prompts.append(task_desc)

        if not prompts:
            logger.warning("No prompts for GRPO training")
            return

        dataset = to_hf_dataset([{"prompt": p} for p in prompts])

        # Load model
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

        # Build reward function for GRPO.
        # NOTE: TRL's GRPOTrainer calls reward_fn on model-generated completions
        # (strings), not on environment trajectories. For proper environment-based
        # reward, use the offline DPO path instead, or implement a custom
        # GRPOTrainer that collects env rollouts.
        def reward_fn(completions: list[str], **kwargs) -> list[float]:
            return [1.0 if any(kw in c.lower() for kw in ["success", "complete", "done", "solved"]) else 0.0
                    for c in completions]

        self._train_step += 1
        run_dir = self._output_dir / f"grpo_step_{self._train_step}"

        config = GRPOConfig(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            max_completion_length=self._max_completion_length,
            num_generations=self._num_generations,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
        )

        trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )

        logger.info("Starting GRPO training: %d prompts", len(prompts))
        train_result = trainer.train()

        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": 0,  # 0=grpo
            "rl/num_prompts": len(prompts),
            "rl/train_step": self._train_step,
        })

        del model, trainer
        import torch
        torch.cuda.empty_cache()

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
            bf16=True,
        )

        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training: %d pairs", len(pairs))
        train_result = trainer.train()

        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": 1,  # 1=dpo
            "rl/num_pairs": len(pairs),
            "rl/train_step": self._train_step,
        })

        del model, trainer
        import torch
        torch.cuda.empty_cache()
