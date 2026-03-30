"""SFT Evolver: supervised fine-tuning on successful trajectories.

Pipeline:
1. Filter trajectories by success/reward threshold
2. Convert to instruction-following format
3. Fine-tune LoRA adapter using TRL SFTTrainer + PEFT
4. Save new checkpoint → hot-swap on vLLM
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.evolution.data.dataset import to_hf_dataset, trajectories_to_sft_data

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


@EVOLVER_REGISTRY.register("sft")
class SFTEvolver(Evolver):
    """Supervised fine-tuning on successful trajectories.

    Works with LoRATarget: produces a new LoRA checkpoint.
    Also works with PromptTarget: collects best prompts from trajectories.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:1",
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 2048,
        reward_threshold: float = 0.0,
        output_dir: str = "outputs/sft",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        lora_config: dict[str, Any] | None = None,
        trainer_callbacks: list | None = None,
        model_init_fn: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._lr = learning_rate
        self._epochs = num_epochs
        self._batch_size = batch_size
        self._grad_accum = gradient_accumulation_steps
        self._max_length = max_length
        self._reward_threshold = reward_threshold
        self._output_dir = Path(output_dir)
        self._torch_dtype = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._trainer_callbacks = trainer_callbacks
        self._model_init_fn = model_init_fn
        self._lora_config = lora_config
        self._train_step = 0

    def requires_trajectories(self) -> bool:
        return True

    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        **kwargs,
    ) -> None:
        # 1. Filter successful trajectories
        good_trajs = [
            t for t in trajectories
            if t.success or t.total_reward > self._reward_threshold
        ]
        if not good_trajs:
            logger.warning("No successful trajectories for SFT (threshold=%.2f)", self._reward_threshold)
            metrics.log({"sft/num_samples": 0})
            return

        logger.info("SFT: %d/%d trajectories pass filter", len(good_trajs), len(trajectories))

        # 2. Convert to SFT format
        sft_data = trajectories_to_sft_data(
            good_trajs,
            system_prompt=agent.brain.system_prompt,
        )
        if not sft_data:
            logger.warning("No SFT samples generated")
            return

        dataset = to_hf_dataset(sft_data)

        # 3. Load trainable model
        from sea.llm.hf_backend import HFTrainingBackend

        hf = HFTrainingBackend(
            model_name=self._model_name,
            device=self._device,
            torch_dtype=self._torch_dtype,
            load_in_4bit=self._load_in_4bit,
        )

        # Get current adapter path from target's evolvable state
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

        # 4. Train with TRL SFTTrainer
        from trl import SFTTrainer, SFTConfig

        self._train_step += 1
        run_dir = self._output_dir / f"step_{self._train_step}"

        training_config = SFTConfig(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            max_length=self._max_length,
            logging_steps=10,
            save_strategy="no",
            bf16=(self._torch_dtype == "bfloat16"),
            fp16=(self._torch_dtype == "float16"),
            remove_unused_columns=False,
        )

        # Apply custom model init if provided (e.g., O-LoRA setup)
        if self._model_init_fn:
            model = self._model_init_fn(model)

        trainer_kwargs: dict[str, Any] = {}
        if self._trainer_callbacks:
            trainer_kwargs["callbacks"] = self._trainer_callbacks

        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            **trainer_kwargs,
        )

        logger.info("Starting SFT training: %d samples, %d epochs", len(dataset), self._epochs)
        train_result = trainer.train()

        # 5. Save adapter
        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)

        # 6. Update target
        target.set_evolvable_state(adapter_path)

        # 7. Hot-swap on inference backend
        agent.brain.swap_lora(str(adapter_path))

        # 8. Log metrics
        metrics.log({
            "sft/num_samples": len(sft_data),
            "sft/train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else 0,
            "sft/train_step": self._train_step,
        })

        # Clean up training model to free GPU memory
        del model, trainer
        import torch
        torch.cuda.empty_cache()

        logger.info("SFT complete: adapter saved to %s", adapter_path)
