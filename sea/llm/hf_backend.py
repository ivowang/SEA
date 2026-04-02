"""HuggingFace backend for training (not inference).

Loads models via transformers + PEFT for gradient-based training with TRL.
Designed for GPU 1 in the 2-GPU setup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HFTrainingBackend:
    """Manages model loading for PEFT-based training.

    This is NOT an LLMBackend -- it exists solely to provide a trainable
    model to SFTEvolver / RLEvolver.  After training, the new LoRA
    checkpoint is saved to disk and hot-swapped into the VLLMBackend.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:1",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit")
        self._model_name = model_name
        self._device = device
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_trainable_model(
        self,
        adapter_path: str | Path | None = None,
        lora_config: dict[str, Any] | None = None,
    ):
        """Load base model with PEFT adapter for training.

        Args:
            adapter_path: Path to existing adapter to continue from.
            lora_config: LoRA config dict if creating a fresh adapter.
                Keys: r, lora_alpha, target_modules, lora_dropout, etc.

        Returns:
            A PeftModel ready for training.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, PeftModel

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._torch_dtype, torch.bfloat16)

        quantization_config = None
        if self._load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self._load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info("Loading base model: %s on %s", self._model_name, self._device)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self._trust_remote_code,
        }

        if quantization_config:
            # Quantized models need device_map for mixed-precision placement
            # Use a single-device map pointing to our training GPU
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = {"": self._device}
        else:
            # Full precision: load to CPU then move to target device
            load_kwargs["device_map"] = {"": self._device}

        model = AutoModelForCausalLM.from_pretrained(self._model_name, **load_kwargs)

        if quantization_config:
            try:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
            except ImportError:
                pass

        if adapter_path is not None:
            logger.info("Loading existing adapter from: %s", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        else:
            defaults = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            if lora_config:
                defaults.update(lora_config)
            logger.info("Creating fresh LoRA adapter: %s", defaults)
            config = LoraConfig(**defaults)
            model = get_peft_model(model, config)
            model.print_trainable_parameters()

        # Enable input gradients for gradient checkpointing compatibility
        model.enable_input_require_grads()

        return model

    def get_tokenizer(self):
        """Load the tokenizer for the base model."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def save_adapter(model, path: str | Path) -> None:
        """Save the PEFT adapter to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))
        logger.info("Saved adapter to: %s", path)
