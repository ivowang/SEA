#!/usr/bin/env python3
"""Standalone worker for parallel trajectory collection.

Each worker runs episodes in a loop, appending results to a shared
JSONL file (one JSON object per line). The main process monitors
the file and kills workers when enough data is collected.

Usage:
    python -m sea.evolution.data.parallel_worker \
        --output /path/to/shared_output.jsonl \
        --env alfworld \
        --env-kwargs '{"split": "eval_out_of_distribution"}' \
        --backend-kwargs '{"model": "...", "base_url": "...", "api_key": "..."}' \
        --system-prompt "..." \
        --max-episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import fcntl
from pathlib import Path

# Prevent CUDA init in API-only workers
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Shared JSONL output file")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--env", default="alfworld")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--backend-type", default="api")
    parser.add_argument("--backend-kwargs", default="{}")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task-type-filter", default=None,
                        help="Only collect episodes of this task type")
    parser.add_argument("--only-successful", action="store_true",
                        help="Only save successful trajectories")
    parser.add_argument("--task-id", default=None,
                        help="Specific task_id to use for every episode reset")
    args = parser.parse_args()

    env_kwargs = json.loads(args.env_kwargs)
    backend_kwargs = json.loads(args.backend_kwargs)

    # Inject task_type_filter into env_kwargs
    if args.task_type_filter:
        env_kwargs["task_type_filter"] = args.task_type_filter

    # Create env — register all benchmark adapters
    from sea.core.registry import ENV_REGISTRY
    import sea.env.benchmarks.textcraft  # noqa
    import sea.env.benchmarks.alfworld  # noqa
    try:
        import sea.env.benchmarks.webshop  # noqa
    except ImportError:
        pass  # WebShop not installed
    env = ENV_REGISTRY.build(args.env, **env_kwargs)

    # Create agent
    from sea.llm.api_backend import APIBackend
    from sea.agent.agent import SEAAgent
    from sea.agent.brain import LLMBrain
    from sea.agent.memory.working import WorkingMemory
    from sea.agent.planner import ReActPlanner

    backend = APIBackend(**backend_kwargs)
    agent = SEAAgent(
        brain=LLMBrain(
            backend=backend,
            system_prompt=args.system_prompt,
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
        ),
        memory=WorkingMemory(max_size=20),
        planner=ReActPlanner(),
    )

    # Run episodes in a loop, append each to JSONL
    output_path = Path(args.output)
    for i in range(args.max_episodes):
        try:
            traj = agent.run_episode(env, task_id=args.task_id)

            # Skip unsuccessful if only_successful is set
            if args.only_successful and not traj.success:
                continue

            record = {
                "task_id": traj.task_id,
                "task_type": traj.task_type,
                "total_reward": traj.total_reward,
                "success": traj.success,
                "num_steps": len(traj),
                "metadata": traj.metadata,
                "steps": [
                    {
                        "observation": s.observation.text[:500],
                        "available_actions": s.observation.available_actions,
                        "action": s.action.text,
                        "action_type": s.action.action_type,
                        "action_metadata": {
                            k: (str(v)[:300] if not isinstance(v, (str, int, float, bool, list)) else v)
                            for k, v in s.action.metadata.items()
                        },
                        "next_observation": s.next_observation.text[:500] if s.next_observation else "",
                        "next_available_actions": s.next_observation.available_actions if s.next_observation else None,
                        "reward": s.reward,
                        "done": s.done,
                        "info": {k: v for k, v in s.info.items() if isinstance(v, (str, int, float, bool))},
                    }
                    for s in traj.steps
                ],
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"

            # Atomic append with file lock — flush before unlock
            with open(output_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN)

        except Exception as e:
            print(f"Episode {i} failed: {e}", file=sys.stderr)

    env.close()


if __name__ == "__main__":
    main()
