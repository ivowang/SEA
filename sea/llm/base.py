"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sea.core.types import GenerationOutput


class LLMBackend(ABC):
    """Unified interface for LLM inference.

    Concrete implementations include:
    - VLLMBackend: local vLLM with LoRA hot-swap (primary)
    - VLLMServerBackend: connects to running vLLM OpenAI-compatible server
    - APIBackend: any OpenAI-compatible API endpoint
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        lora_name: str | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate a single completion from a message list."""
        ...

    @abstractmethod
    def generate_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        lora_name: str | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate completions for a batch of message lists."""
        ...

    @abstractmethod
    def load_lora(self, path: str, name: str | None = None) -> None:
        """Load a LoRA adapter. Hot-swap if the backend supports it."""
        ...

    @abstractmethod
    def unload_lora(self, name: str) -> None:
        """Unload a previously loaded LoRA adapter."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the base model identifier."""
        ...

    def supports_lora(self) -> bool:
        """Whether this backend supports LoRA adapter loading."""
        return False

    def list_loras(self) -> list[str]:
        """Return names of currently loaded LoRA adapters."""
        return []
