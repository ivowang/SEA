#!/usr/bin/env python3
"""Standalone worker script for parallel trajectory collection.

Usage:
    python -m sea.evolution.data.parallel_worker \
        --output /tmp/traj_worker_0.json \
        --n 5 \
        --env alfworld \
        --env-kwargs '{"split": "eval_out_of_distribution", "max_steps_val": 30}' \
        --backend-type api \
        --backend-kwargs '{"model": "openai/gpt-5.4-nano", "base_url": "...", "api_key": "..."}' \
        --system-prompt "You are a household robot."

Each invocation creates its own env + agent, runs n episodes,
and writes results to --output as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    # Prevent CUDA init in API-only workers (saves GPU memory)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--env", default="alfworld")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--backend-type", default="api")
    parser.add_argument("--backend-kwargs", default="{}")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    env_kwargs = json.loads(args.env_kwargs)
    backend_kwargs = json.loads(args.backend_kwargs)

    # Create env
    from sea.core.registry import ENV_REGISTRY
    import sea.env.benchmarks.textcraft  # noqa
    import sea.env.benchmarks.alfworld  # noqa
    env = ENV_REGISTRY.build(args.env, **env_kwargs)

    # Create agent
    from sea.llm.api_backend import APIBackend
    from sea.agent.agent import SEAAgent
    from sea.agent.brain import LLMBrain
    from sea.agent.memory.episodic import EpisodicMemory
    from sea.agent.planner import ReActPlanner

    backend = APIBackend(**backend_kwargs)
    agent = SEAAgent(
        brain=LLMBrain(
            backend=backend,
            system_prompt=args.system_prompt,
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
        ),
        memory=EpisodicMemory(max_size=20),
        planner=ReActPlanner(),
    )

    # Collect
    results = []
    for i in range(args.n):
        try:
            traj = agent.run_episode(env)
            results.append({
                "task_id": traj.task_id,
                "task_type": traj.task_type,
                "total_reward": traj.total_reward,
                "success": traj.success,
                "num_steps": len(traj),
                "metadata": traj.metadata,
                "steps": [
                    {
                        "observation": s.observation.text[:500],
                        "action": s.action.text,
                        "action_type": s.action.action_type,
                        "action_metadata": {k: str(v)[:200] for k, v in s.action.metadata.items()},
                        "next_observation": s.next_observation.text[:500] if s.next_observation else "",
                        "reward": s.reward,
                        "done": s.done,
                        "info": {k: v for k, v in s.info.items() if isinstance(v, (str, int, float, bool))},
                    }
                    for s in traj.steps
                ],
            })
        except Exception as e:
            print(f"Episode {i} failed: {e}", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False))
    print(f"Worker done: {len(results)} trajectories saved to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
