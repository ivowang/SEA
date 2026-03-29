"""LoRA adapter as an evolution target.

Manages LoRA checkpoint creation, saving, and hot-swapping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sea.core.base import Evolvable

logger = logging.getLogger(__name__)


class LoRATarget(Evolvable[Path]):
    """Wraps a LoRA adapter as an evolution target.

    The evolvable state is the Path to the adapter checkpoint directory.
    When evolved (SFT/RL produces a new checkpoint), the new path is set
    and the caller is responsible for hot-swapping on the inference backend.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_dir: Path | str,
        lora_config: dict[str, Any] | None = None,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_dir = Path(adapter_dir)
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self._version = 0
        # Multi-adapter tracking for continual learning
        self.adapter_history: dict[str, Path] = {}  # task_type -> adapter path
        self.r_sum: int = 0  # accumulated rank for O-LoRA

    @property
    def current_path(self) -> Path:
        return self.adapter_dir

    @property
    def version(self) -> int:
        return self._version

    def create_fresh_adapter(self, output_dir: Path | None = None) -> Path:
        """Initialize a new LoRA adapter using PEFT and save it.

        Returns:
            Path to the saved adapter.
        """
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        import torch

        output_dir = output_dir or self.adapter_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Creating fresh LoRA adapter for %s", self.base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        config = LoraConfig(**self.lora_config)
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(str(output_dir))
        del model, peft_model

        self.adapter_dir = output_dir
        logger.info("Saved fresh adapter to %s", output_dir)
        return output_dir

    # -- Evolvable[Path] --

    def get_evolvable_state(self) -> Path:
        return self.adapter_dir

    def set_evolvable_state(self, state: Path) -> None:
        self.adapter_dir = Path(state)
        self._version += 1
        logger.info("LoRA target updated to %s (version %d)", self.adapter_dir, self._version)

    def register_task_adapter(self, task_type: str, adapter_path: Path) -> None:
        """Record an adapter checkpoint for a specific task type (continual learning)."""
        self.adapter_history[task_type] = adapter_path
        logger.info("Registered adapter for task '%s': %s", task_type, adapter_path)

    def get_task_adapter(self, task_type: str) -> Path | None:
        """Get the adapter path for a specific task type."""
        return self.adapter_history.get(task_type)

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "lora_adapter",
            "base_model": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "version": self._version,
            "lora_config": self.lora_config,
            "r_sum": self.r_sum,
            "task_history": list(self.adapter_history.keys()),
        }

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "base_model_name": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "lora_config": self.lora_config,
            "version": self._version,
        }
        (path / "lora_target.json").write_text(json.dumps(state, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "lora_target.json"
        if fp.exists():
            state = json.loads(fp.read_text())
            self.base_model_name = state["base_model_name"]
            self.adapter_dir = Path(state["adapter_dir"])
            self.lora_config = state["lora_config"]
            self._version = state["version"]

    def state_dict(self) -> dict[str, Any]:
        return self.evolution_metadata()
