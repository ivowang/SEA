"""Trajectory collection and buffering."""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, Callable

from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """Ring buffer of trajectories with filtering and sampling."""

    def __init__(self, max_size: int = 10000) -> None:
        self._buffer: deque[Trajectory] = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory) -> None:
        self._buffer.append(trajectory)

    def add_batch(self, trajectories: list[Trajectory]) -> None:
        for t in trajectories:
            self._buffer.append(t)

    def sample(
        self,
        n: int,
        filter_fn: Callable[[Trajectory], bool] | None = None,
    ) -> list[Trajectory]:
        """Sample n trajectories, optionally filtered."""
        candidates = list(self._buffer)
        if filter_fn:
            candidates = [t for t in candidates if filter_fn(t)]
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    def successful(self, threshold: float = 0.0) -> list[Trajectory]:
        """Return trajectories with total_reward > threshold or success=True."""
        return [
            t for t in self._buffer
            if t.success or t.total_reward > threshold
        ]

    def failed(self) -> list[Trajectory]:
        """Return failed trajectories."""
        return [t for t in self._buffer if not t.success and t.total_reward <= 0]

    def all(self) -> list[Trajectory]:
        return list(self._buffer)

    def by_task_type(self, task_type: str) -> list[Trajectory]:
        """Return trajectories of a specific task type."""
        return [t for t in self._buffer if t.task_type == task_type]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict[str, float]:
        if not self._buffer:
            return {"size": 0, "success_rate": 0.0, "avg_reward": 0.0}
        successes = sum(1 for t in self._buffer if t.success)
        avg_reward = sum(t.total_reward for t in self._buffer) / len(self._buffer)
        return {
            "size": len(self._buffer),
            "success_rate": successes / len(self._buffer),
            "avg_reward": avg_reward,
            "avg_steps": sum(len(t) for t in self._buffer) / len(self._buffer),
        }


class TrajectoryCollector:
    """Collects trajectories from agent-environment interaction."""

    def __init__(self, buffer: TrajectoryBuffer | None = None) -> None:
        self.buffer = buffer or TrajectoryBuffer()

    def collect(
        self,
        agent: SEAAgent,
        envs: list[SEAEnv],
        n: int,
        task_ids: list[str] | None = None,
        task_type_filter: str | None = None,
    ) -> list[Trajectory]:
        """Collect n trajectories across the given environments.

        Args:
            agent: Agent to collect with.
            envs: Environments to interact with.
            n: Number of trajectories to collect.
            task_ids: Optional specific task IDs to use.
            task_type_filter: If set, only collect from tasks of this type.
                Requires the env's reset() to return task_type in info dict.
        """
        trajectories: list[Trajectory] = []

        # Build task list
        if task_ids is None:
            all_tasks = []
            for env in envs:
                all_tasks.extend(
                    [(env, tid) for tid in env.get_task_ids()]
                )
            random.shuffle(all_tasks)
        else:
            all_tasks = [(random.choice(envs), tid) for tid in task_ids]

        for i in range(n):
            env, task_id = all_tasks[i % len(all_tasks)]
            try:
                traj = agent.run_episode(env, task_id=task_id)
                trajectories.append(traj)
                self.buffer.add(traj)
            except Exception as e:
                logger.error("Failed to collect trajectory: %s", e)

        logger.info(
            "Collected %d trajectories (success rate: %.1f%%)",
            len(trajectories),
            100 * sum(1 for t in trajectories if t.success) / max(len(trajectories), 1),
        )
        return trajectories
