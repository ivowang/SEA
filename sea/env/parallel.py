"""Parallel environment runner for batch trajectory collection."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable

from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


class ParallelEnvRunner:
    """Runs multiple environments in parallel for batch trajectory collection.

    Uses ThreadPoolExecutor — suitable for environments that are I/O-bound
    (most text environments block on LLM generation, not CPU).
    """

    def __init__(
        self,
        env_factory: Callable[[], SEAEnv],
        num_envs: int = 4,
    ) -> None:
        self.envs = [env_factory() for _ in range(num_envs)]
        self._num_envs = num_envs

    def collect_trajectories(
        self,
        agent: SEAAgent,
        n: int,
        task_ids: list[str] | None = None,
    ) -> list[Trajectory]:
        """Collect *n* trajectories in parallel.

        Args:
            agent: The agent to run episodes with.
            n: Number of trajectories to collect.
            task_ids: Optional list of task IDs. Cycled if shorter than n.

        Returns:
            List of collected trajectories.
        """
        trajectories: list[Trajectory] = []
        remaining = n
        batch = 0

        while remaining > 0:
            batch_size = min(remaining, self._num_envs)
            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = []
                for i in range(batch_size):
                    env = self.envs[i % self._num_envs]
                    task_id = None
                    if task_ids:
                        idx = (batch * self._num_envs + i) % len(task_ids)
                        task_id = task_ids[idx]
                    futures.append(pool.submit(agent.run_episode, env, task_id=task_id))

                for future in as_completed(futures):
                    try:
                        traj = future.result()
                        trajectories.append(traj)
                    except Exception as e:
                        logger.error("Episode failed: %s", e)

            remaining -= batch_size
            batch += 1

        logger.info("Collected %d trajectories", len(trajectories))
        return trajectories

    def close(self) -> None:
        for env in self.envs:
            env.close()
