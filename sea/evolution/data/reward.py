"""Reward functions for evaluating trajectories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sea.core.registry import REWARD_REGISTRY
from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.llm.base import LLMBackend


class RewardFunction(ABC):
    """Computes reward for a trajectory."""

    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> float:
        ...


@REWARD_REGISTRY.register("environment")
class EnvironmentReward(RewardFunction):
    """Uses the environment's native reward signal."""

    def __call__(self, trajectory: Trajectory) -> float:
        return trajectory.total_reward


@REWARD_REGISTRY.register("success")
class SuccessReward(RewardFunction):
    """Binary reward: 1.0 if successful, 0.0 otherwise."""

    def __call__(self, trajectory: Trajectory) -> float:
        return 1.0 if trajectory.success else 0.0


@REWARD_REGISTRY.register("step_penalty")
class StepPenaltyReward(RewardFunction):
    """Penalises long episodes to encourage efficiency."""

    def __init__(self, success_reward: float = 1.0, step_penalty: float = 0.01) -> None:
        self._success_reward = success_reward
        self._step_penalty = step_penalty

    def __call__(self, trajectory: Trajectory) -> float:
        base = self._success_reward if trajectory.success else 0.0
        penalty = self._step_penalty * len(trajectory)
        return base - penalty


@REWARD_REGISTRY.register("llm_judge")
class LLMJudgeReward(RewardFunction):
    """Uses an LLM to judge trajectory quality."""

    def __init__(self, backend: LLMBackend, rubric: str = "") -> None:
        self._backend = backend
        self._rubric = rubric or (
            "Rate the quality of the agent's trajectory on a scale of 0.0 to 1.0. "
            "Consider: task completion, efficiency, and correctness. "
            "Reply with ONLY a number between 0.0 and 1.0."
        )

    def __call__(self, trajectory: Trajectory) -> float:
        steps_text = "\n".join(
            f"Step {i}: Action={s.action.text}, Obs={s.observation.text[:100]}"
            for i, s in enumerate(trajectory.steps[-10:])  # Last 10 steps
        )
        messages = [
            {"role": "system", "content": self._rubric},
            {"role": "user", "content": f"Trajectory (task: {trajectory.task_id}):\n{steps_text}"},
        ]
        output = self._backend.generate(messages, temperature=0.0, max_tokens=50)
        # Extract first float from response
        import re
        match = re.search(r"(\d+\.?\d*)", output.text)
        if match:
            value = float(match.group(1))
            return min(max(value, 0.0), 1.0)  # clamp to [0, 1]
        return 0.0


@REWARD_REGISTRY.register("composite")
class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward functions."""

    def __init__(self, rewards: list[tuple[RewardFunction, float]]) -> None:
        self._rewards = rewards

    def __call__(self, trajectory: Trajectory) -> float:
        return sum(w * r(trajectory) for r, w in self._rewards)
