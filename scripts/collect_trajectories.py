#!/usr/bin/env python3
"""Collect trajectories without training — useful for building datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sea.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="SEA: Collect trajectories")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--output", type=str, default="outputs/trajectories.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    from sea.utils.config import load_config
    from scripts.run_evolution import build_from_config

    cfg = load_config(args.config)
    pipeline = build_from_config(cfg)

    logger.info("Collecting %d trajectories...", args.num_trajectories)
    trajectories = pipeline.collector.collect(
        pipeline.agent,
        pipeline.envs,
        n=args.num_trajectories,
    )

    # Save trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for traj in trajectories:
        data.append({
            "task_id": traj.task_id,
            "total_reward": traj.total_reward,
            "success": traj.success,
            "num_steps": len(traj),
            "steps": [
                {
                    "observation": s.observation.text[:500],
                    "action": s.action.text,
                    "reward": s.reward,
                }
                for s in traj.steps
            ],
        })

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved %d trajectories to %s", len(data), output_path)

    # Summary
    success_rate = sum(1 for t in trajectories if t.success) / max(len(trajectories), 1)
    avg_reward = sum(t.total_reward for t in trajectories) / max(len(trajectories), 1)
    logger.info("Success rate: %.2f%%", success_rate * 100)
    logger.info("Avg reward: %.4f", avg_reward)


if __name__ == "__main__":
    main()
