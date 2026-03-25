#!/usr/bin/env python3
"""Evaluate a saved agent checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sea.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="SEA: Evaluate agent checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    from sea.utils.config import load_config
    from scripts.run_evolution import build_from_config

    cfg = load_config(args.config)
    pipeline = build_from_config(cfg)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    agent_ckpt = ckpt_path / "agent" if (ckpt_path / "agent").exists() else ckpt_path
    pipeline.agent.load_checkpoint(agent_ckpt)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    # Run evaluation
    pipeline.evaluator._num_episodes = args.num_episodes
    results = pipeline.evaluator.evaluate(pipeline.agent, pipeline.envs)

    logger.info("Evaluation results:")
    logger.info("  Success rate: %.2f%%", results.success_rate * 100)
    logger.info("  Avg reward:   %.4f", results.avg_reward)
    logger.info("  Avg steps:    %.1f", results.avg_steps)
    logger.info("  Episodes:     %d", results.num_episodes)
    for env_name, rate in results.per_env.items():
        logger.info("  %s success:  %.2f%%", env_name, rate * 100)


if __name__ == "__main__":
    main()
