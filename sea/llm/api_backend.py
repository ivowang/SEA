"""OpenAI-compatible API backend for remote LLM inference."""

from __future__ import annotations

import logging
from typing import Any

from sea.core.registry import LLM_BACKEND_REGISTRY
from sea.core.types import GenerationOutput
from sea.llm.base import LLMBackend

logger = logging.getLogger(__name__)


@LLM_BACKEND_REGISTRY.register("api")
class APIBackend(LLMBackend):
    """Calls any OpenAI-compatible chat completions endpoint.

    Useful as a fallback or for using commercial APIs during development.
    Does not support LoRA operations.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package required. Install with: pip install openai") from e

        self._model_name = model
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

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
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        choice = response.choices[0]
        return GenerationOutput(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

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
        return [
            self.generate(msgs, max_tokens=max_tokens, temperature=temperature,
                          top_p=top_p, stop=stop)
            for msgs in message_batches
        ]

    def load_lora(self, path: str, name: str | None = None) -> None:
        raise NotImplementedError("API backend does not support LoRA")

    def unload_lora(self, name: str) -> None:
        raise NotImplementedError("API backend does not support LoRA")
