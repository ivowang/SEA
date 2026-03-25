"""vLLM backend with LoRA hot-swapping support.

Provides two modes:
- VLLMBackend: in-process vLLM engine (single-machine research)
- VLLMServerBackend: connects to a running vLLM server (multi-process setups)
"""

from __future__ import annotations

import logging
from typing import Any

from sea.core.registry import LLM_BACKEND_REGISTRY
from sea.core.types import GenerationOutput
from sea.llm.base import LLMBackend

logger = logging.getLogger(__name__)


@LLM_BACKEND_REGISTRY.register("vllm")
class VLLMBackend(LLMBackend):
    """In-process vLLM engine with LoRA hot-swap.

    Designed for GPU 0 in the 2-GPU setup.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        enable_lora: bool = True,
        max_lora_rank: int = 64,
        max_loras: int = 4,
        dtype: str = "auto",
        device: str = "cuda:0",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            import os
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":")[-1])

            from vllm import LLM, SamplingParams  # noqa: F401
            from vllm.lora.request import LoRARequest  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm is required for VLLMBackend. "
                "Install with: pip install 'vllm>=0.18.0'"
            ) from e

        self._model_name = model
        self._enable_lora = enable_lora

        logger.info("Initializing vLLM engine: model=%s, tp=%d", model, tensor_parallel_size)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank if enable_lora else None,
            max_loras=max_loras if enable_lora else None,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        self._lora_counter = 0
        self._active_loras: dict[str, Any] = {}  # name -> LoRARequest
        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_lora(self) -> bool:
        return self._enable_lora

    def list_loras(self) -> list[str]:
        return list(self._active_loras.keys())

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Apply chat template to messages."""
        tokenizer = self.llm.get_tokenizer()
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_output(self, output) -> GenerationOutput:
        """Convert vLLM output to GenerationOutput."""
        completion = output.outputs[0]
        return GenerationOutput(
            text=completion.text,
            token_ids=list(completion.token_ids) if completion.token_ids else [],
            logprobs=(
                [lp.logprob for lp in completion.logprobs]
                if completion.logprobs
                else None
            ),
            prompt_token_ids=list(output.prompt_token_ids) if output.prompt_token_ids else [],
            finish_reason=completion.finish_reason or "",
            usage={
                "prompt_tokens": len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                "completion_tokens": len(completion.token_ids) if completion.token_ids else 0,
            },
        )

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
        prompt = self._build_prompt(messages)
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        lora_request = self._active_loras.get(lora_name) if lora_name else None
        outputs = self.llm.generate([prompt], params, lora_request=lora_request)
        return self._parse_output(outputs[0])

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
        prompts = [self._build_prompt(msgs) for msgs in message_batches]
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        lora_request = self._active_loras.get(lora_name) if lora_name else None
        outputs = self.llm.generate(prompts, params, lora_request=lora_request)
        return [self._parse_output(o) for o in outputs]

    def load_lora(self, path: str, name: str | None = None) -> None:
        if not self._enable_lora:
            raise RuntimeError("LoRA not enabled on this backend")
        self._lora_counter += 1
        name = name or f"lora_{self._lora_counter}"
        self._active_loras[name] = self._LoRARequest(
            lora_name=name,
            lora_int_id=self._lora_counter,
            lora_path=path,
        )
        logger.info("Loaded LoRA adapter '%s' from %s", name, path)

    def unload_lora(self, name: str) -> None:
        if name in self._active_loras:
            del self._active_loras[name]
            logger.info("Unloaded LoRA adapter '%s'", name)


@LLM_BACKEND_REGISTRY.register("vllm_server")
class VLLMServerBackend(LLMBackend):
    """Connects to a running vLLM OpenAI-compatible server.

    LoRA hot-swap via POST /v1/load_lora_adapter endpoint.
    Start server with: VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "default",
        api_key: str = "unused",
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package required for VLLMServerBackend. "
                "Install with: pip install openai"
            ) from e

        self._model_name = model
        self._base_url = base_url
        self.client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        self._loaded_loras: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_lora(self) -> bool:
        return True

    def list_loras(self) -> list[str]:
        return list(self._loaded_loras)

    def _parse_chat_output(self, response) -> GenerationOutput:
        choice = response.choices[0]
        return GenerationOutput(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

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
        model = lora_name if lora_name else self._model_name
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        return self._parse_chat_output(response)

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
            self.generate(
                msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                lora_name=lora_name,
            )
            for msgs in message_batches
        ]

    def load_lora(self, path: str, name: str | None = None) -> None:
        import httpx

        name = name or "default_lora"
        resp = httpx.post(
            f"{self._base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": path},
            timeout=60.0,
        )
        resp.raise_for_status()
        if name not in self._loaded_loras:
            self._loaded_loras.append(name)
        logger.info("Loaded LoRA '%s' on server from %s", name, path)

    def unload_lora(self, name: str) -> None:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=60.0,
        )
        resp.raise_for_status()
        if name in self._loaded_loras:
            self._loaded_loras.remove(name)
        logger.info("Unloaded LoRA '%s' from server", name)
