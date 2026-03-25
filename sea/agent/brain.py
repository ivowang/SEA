"""LLMBrain: wraps an LLM backend and manages agent-level LLM state.

The brain is Evolvable — its LoRA adapter and system prompt can both
be modified by evolution methods.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sea.core.base import Evolvable
from sea.core.types import GenerationOutput
from sea.llm.base import LLMBackend

logger = logging.getLogger(__name__)


class LLMBrain(Evolvable[dict[str, Any]]):
    """The agent's LLM-based reasoning core.

    Wraps an LLMBackend and adds agent-specific state:
    - system_prompt: can be evolved by prompt evolvers
    - active LoRA adapter name: can be evolved by SFT/RL evolvers
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt: str = "",
        lora_name: str | None = None,
        lora_path: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
    ) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.lora_name = lora_name
        self.lora_path = lora_path
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        # Load LoRA if specified
        if lora_path and backend.supports_lora():
            backend.load_lora(lora_path, name=lora_name)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate a response from the LLM."""
        return self.backend.generate(
            messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            stop=stop,
            lora_name=self.lora_name,
            **kwargs,
        )

    def generate_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate responses for a batch of message lists."""
        return self.backend.generate_batch(
            message_batches,
            temperature=kwargs.pop("temperature", self.default_temperature),
            max_tokens=kwargs.pop("max_tokens", self.default_max_tokens),
            lora_name=self.lora_name,
            **kwargs,
        )

    def swap_lora(self, path: str, name: str | None = None) -> None:
        """Hot-swap to a new LoRA adapter."""
        if not self.backend.supports_lora():
            raise RuntimeError(f"Backend {type(self.backend).__name__} does not support LoRA")
        name = name or "default"
        self.backend.load_lora(path, name=name)
        self.lora_name = name
        self.lora_path = path
        logger.info("Hot-swapped LoRA to '%s' from %s", name, path)

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "system_prompt": self.system_prompt,
            "lora_name": self.lora_name,
            "lora_path": self.lora_path,
            "model_name": self.backend.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }
        (path / "brain.json").write_text(json.dumps(state, indent=2))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "brain.json"
        if not fp.exists():
            return
        state = json.loads(fp.read_text())
        self.system_prompt = state.get("system_prompt", self.system_prompt)
        self.default_temperature = state.get("default_temperature", self.default_temperature)
        self.default_max_tokens = state.get("default_max_tokens", self.default_max_tokens)
        lora_path = state.get("lora_path")
        if lora_path and self.backend.supports_lora():
            self.swap_lora(lora_path, name=state.get("lora_name"))

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.backend.model_name,
            "system_prompt": self.system_prompt[:200],
            "lora_name": self.lora_name,
            "lora_path": self.lora_path,
        }

    # -- Evolvable --

    def get_evolvable_state(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "lora_name": self.lora_name,
            "lora_path": self.lora_path,
        }

    def set_evolvable_state(self, state: dict[str, Any]) -> None:
        if "system_prompt" in state:
            self.system_prompt = state["system_prompt"]
        new_lora = state.get("lora_path")
        if new_lora and new_lora != self.lora_path:
            self.swap_lora(new_lora, name=state.get("lora_name"))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "llm_brain",
            "model_name": self.backend.model_name,
            "has_lora": self.lora_name is not None,
            "prompt_length": len(self.system_prompt),
        }
