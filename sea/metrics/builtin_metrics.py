"""Built-in metric functions for evaluating agent performance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent


def success_rate(trajectories: list[Trajectory]) -> float:
    """Fraction of successful trajectories."""
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.success) / len(trajectories)


def avg_reward(trajectories: list[Trajectory]) -> float:
    """Mean total reward across trajectories."""
    if not trajectories:
        return 0.0
    return sum(t.total_reward for t in trajectories) / len(trajectories)


def avg_episode_length(trajectories: list[Trajectory]) -> float:
    """Mean number of steps per episode."""
    if not trajectories:
        return 0.0
    return sum(len(t) for t in trajectories) / len(trajectories)


def reward_improvement(
    current: list[Trajectory],
    previous: list[Trajectory],
) -> float:
    """Improvement in average reward between two batches."""
    return avg_reward(current) - avg_reward(previous)


def skill_library_size(agent: SEAAgent) -> int:
    """Number of skills in the agent's library."""
    return len(agent.skill_library) if agent.skill_library else 0


def memory_utilization(agent: SEAAgent) -> dict[str, int]:
    """Memory sizes by component."""
    return {"memory_size": agent.memory.size()}
