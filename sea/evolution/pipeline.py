"""Evolution Pipeline: orchestrates collect → evolve → evaluate → repeat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.core.base import Evolvable
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.evolution.data.trajectory import TrajectoryBuffer, TrajectoryCollector

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv
    from sea.metrics.evaluator import Evaluator
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution pipeline."""

    num_iterations: int = 100
    trajectories_per_iteration: int = 64
    eval_every: int = 10
    checkpoint_every: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    log_every: int = 1


class EvolutionPipeline:
    """The main evolution loop.

    Orchestrates: collect trajectories → evolve targets → evaluate → repeat.
    Supports multiple evolvers running on multiple targets simultaneously.
    """

    def __init__(
        self,
        agent: SEAAgent,
        envs: list[SEAEnv],
        evolvers: list[tuple[Evolver, str]],  # (evolver, target_component_name)
        evaluator: Evaluator,
        metrics: MetricsTracker,
        config: EvolutionConfig | None = None,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.evolvers = evolvers
        self.evaluator = evaluator
        self.metrics = metrics
        self.config = config or EvolutionConfig()
        self.buffer = TrajectoryBuffer()
        self.collector = TrajectoryCollector(buffer=self.buffer)
        self._checkpoint_dir = Path(self.config.checkpoint_dir)

    def run(self) -> None:
        """Main evolution loop."""
        logger.info(
            "Starting evolution: %d iterations, %d trajectories/iter, %d evolvers",
            self.config.num_iterations,
            self.config.trajectories_per_iteration,
            len(self.evolvers),
        )

        for iteration in range(self.config.num_iterations):
            iter_start = time.time()
            self.metrics.global_step = iteration

            # 1. Collect trajectories
            logger.info("Iteration %d: collecting trajectories...", iteration)
            trajectories = self.collector.collect(
                self.agent,
                self.envs,
                n=self.config.trajectories_per_iteration,
            )

            # Log collection stats
            if trajectories and iteration % self.config.log_every == 0:
                success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
                avg_reward = sum(t.total_reward for t in trajectories) / len(trajectories)
                avg_steps = sum(len(t) for t in trajectories) / len(trajectories)
                self.metrics.log({
                    "collect/success_rate": success_rate,
                    "collect/avg_reward": avg_reward,
                    "collect/avg_steps": avg_steps,
                    "collect/num_trajectories": len(trajectories),
                })

            # 2. Evolve each target
            evolvable_components = self.agent.evolvable_components()
            for evolver, target_name in self.evolvers:
                if target_name not in evolvable_components:
                    logger.warning("Target '%s' not found in agent's evolvable components", target_name)
                    continue
                target = evolvable_components[target_name]

                logger.info(
                    "Iteration %d: evolving '%s' with %s...",
                    iteration, target_name, type(evolver).__name__,
                )
                try:
                    evolver.evolve(self.agent, target, trajectories, self.metrics,
                                   envs=self.envs)
                except Exception as e:
                    logger.error("Evolution failed for '%s': %s", target_name, e, exc_info=True)

            # 3. Periodic evaluation
            if iteration % self.config.eval_every == 0:
                logger.info("Iteration %d: evaluating...", iteration)
                eval_results = self.evaluator.evaluate(self.agent, self.envs)
                self.metrics.log_eval(eval_results, step=iteration)

            # 4. Periodic checkpoint
            if iteration % self.config.checkpoint_every == 0:
                self._save_checkpoint(iteration)

            iter_time = time.time() - iter_start
            self.metrics.log({"time/iteration_seconds": iter_time})
            logger.info("Iteration %d complete (%.1fs)", iteration, iter_time)

        # Final evaluation
        logger.info("Evolution complete. Running final evaluation...")
        final_results = self.evaluator.evaluate(self.agent, self.envs)
        self.metrics.log_eval(final_results, step=self.config.num_iterations)
        self._save_checkpoint(self.config.num_iterations)

        logger.info(
            "Final results: success_rate=%.2f, avg_reward=%.2f",
            final_results.success_rate,
            final_results.avg_reward,
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """Save agent and evolver states."""
        ckpt_dir = self._checkpoint_dir / f"iter_{iteration}"
        try:
            self.agent.save_checkpoint(ckpt_dir / "agent")
            for evolver, name in self.evolvers:
                evolver.save_checkpoint(ckpt_dir / f"evolver_{name}")
            logger.info("Checkpoint saved: %s", ckpt_dir)
        except Exception as e:
            logger.error("Checkpoint failed: %s", e)
