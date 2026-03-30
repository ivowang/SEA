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
            "Reply with ONLY a single decimal number between 0.0 and 1.0, "
            "for example: 0.75"
        )

    def __call__(self, trajectory: Trajectory) -> float:
        steps_text = "\n".join(
            f"Step {i}: Action={s.action.text}, Obs={s.observation.text[:100]}"
            for i, s in enumerate(trajectory.steps[-10:])
        )
        messages = [
            {"role": "system", "content": self._rubric},
            {"role": "user", "content": f"Trajectory (task: {trajectory.task_id}):\n{steps_text}"},
        ]
        output = self._backend.generate(messages, temperature=0.0, max_tokens=50)

        import re
        text = output.text.strip()

        # Try strict decimal match first (e.g., "0.75", "1.0")
        strict = re.match(r"^(0?\.\d+|1\.0|0|1)$", text)
        if strict:
            return float(strict.group(1))

        # Handle fractions like "7/10"
        frac = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if frac:
            num, den = float(frac.group(1)), float(frac.group(2))
            if den > 0:
                return min(max(num / den, 0.0), 1.0)

        # Handle percentages like "85%"
        pct = re.search(r"(\d+)%", text)
        if pct:
            return min(max(float(pct.group(1)) / 100.0, 0.0), 1.0)

        # Fallback: first decimal-looking number
        fallback = re.search(r"(\d+\.?\d*)", text)
        if fallback:
            value = float(fallback.group(1))
            if value > 1.0:
                value = value / 10.0 if value <= 10 else value / 100.0
            return min(max(value, 0.0), 1.0)

        return 0.0


@REWARD_REGISTRY.register("composite")
class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward functions."""

    def __init__(self, rewards: list[tuple[RewardFunction, float]]) -> None:
        self._rewards = rewards

    def __call__(self, trajectory: Trajectory) -> float:
        return sum(w * r(trajectory) for r, w in self._rewards)
