#!/usr/bin/env python3
"""Start a vLLM server for a model (OpenAI-compatible API)."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="SEA: Start vLLM server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enable-lora", action="store_true", default=True)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
    ]

    if args.enable_lora:
        cmd.extend(["--enable-lora", "--max-lora-rank", str(args.max_lora_rank)])

    env = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.run(cmd, env={**__import__("os").environ, **env})


if __name__ == "__main__":
    main()
