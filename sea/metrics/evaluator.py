"""Evaluator: runs evaluation episodes and computes metrics."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    num_episodes: int = 0
    per_env: dict[str, float] = field(default_factory=dict)
    per_task: dict[str, float] = field(default_factory=dict)
    trajectories: list[Trajectory] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        d = {
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_steps": self.avg_steps,
            "num_episodes": self.num_episodes,
        }
        for env_name, rate in self.per_env.items():
            d[f"{env_name}_success_rate"] = rate
        return d


class Evaluator:
    """Runs evaluation episodes and computes aggregate metrics."""

    def __init__(
        self,
        num_episodes_per_env: int = 20,
        eval_temperature: float = 0.0,  # Greedy for eval
        max_steps: int | None = None,
        eval_seed: int = 42,  # Fixed seed for reproducible task selection
    ) -> None:
        self._num_episodes = num_episodes_per_env
        self._eval_temp = eval_temperature
        self._max_steps = max_steps
        self._eval_seed = eval_seed

    def evaluate(
        self,
        agent: SEAAgent,
        envs: list[SEAEnv],
        task_ids: list[str] | None = None,
    ) -> EvalResults:
        """Run eval episodes across all envs, return aggregated results."""
        all_trajectories: list[Trajectory] = []
        per_env_results: dict[str, list[Trajectory]] = {}

        # Set greedy temperature for eval
        orig_temp = agent.brain.default_temperature
        agent.brain.default_temperature = self._eval_temp

        try:
            for env_idx, env in enumerate(envs):
                env_trajs: list[Trajectory] = []
                available_tasks = task_ids or env.get_task_ids()

                # Determine number of eval episodes
                n_episodes = self._num_episodes
                if available_tasks:
                    n_episodes = min(n_episodes, len(available_tasks))

                # Fixed seed for reproducible task selection
                rng = random.Random(self._eval_seed)
                if available_tasks and n_episodes <= len(available_tasks):
                    selected = rng.sample(available_tasks, n_episodes)
                else:
                    # Sequential env (ALFWorld) or no task IDs — just run N episodes
                    selected = [None] * n_episodes

                n_attempted = 0
                n_errors = 0
                for task_id in selected:
                    n_attempted += 1
                    try:
                        traj = agent.run_episode(
                            env,
                            task_id=task_id,
                            max_steps=self._max_steps,
                        )
                        env_trajs.append(traj)
                    except Exception as e:
                        n_errors += 1
                        logger.error("Eval episode failed (env=%s, task=%s): %s",
                                     env.name, task_id, e)
                if n_errors:
                    logger.warning(
                        "Eval: %d/%d episodes failed for env=%s",
                        n_errors, n_attempted, env.name,
                    )

                # Unique key to avoid collision with multiple envs of same name
                env_key = f"{env.name}_{env_idx}" if len(envs) > 1 else env.name
                per_env_results[env_key] = env_trajs
                all_trajectories.extend(env_trajs)
        finally:
            agent.brain.default_temperature = orig_temp

        return self._aggregate(all_trajectories, per_env_results)

    def _aggregate(
        self,
        trajectories: list[Trajectory],
        per_env: dict[str, list[Trajectory]],
    ) -> EvalResults:
        if not trajectories:
            return EvalResults()

        successes = sum(1 for t in trajectories if t.success)
        return EvalResults(
            success_rate=successes / len(trajectories),
            avg_reward=sum(t.total_reward for t in trajectories) / len(trajectories),
            avg_steps=sum(len(t) for t in trajectories) / len(trajectories),
            num_episodes=len(trajectories),
            per_env={
                name: sum(1 for t in trajs if t.success) / max(len(trajs), 1)
                for name, trajs in per_env.items()
            },
            per_task={
                t.task_id: t.total_reward for t in trajectories
            },
            trajectories=trajectories,
        )
