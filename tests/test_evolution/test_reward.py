"""Tests for reward functions."""

from sea.core.types import Action, Observation, Step, Trajectory
from sea.evolution.data.reward import EnvironmentReward, SuccessReward, StepPenaltyReward, CompositeReward


def _make_traj(reward: float, success: bool, n_steps: int = 3) -> Trajectory:
    return Trajectory(
        steps=[Step(observation=Observation(text=""), action=Action(text=""), reward=0) for _ in range(n_steps)],
        total_reward=reward,
        success=success,
    )


def test_environment_reward():
    r = EnvironmentReward()
    assert r(_make_traj(1.5, True)) == 1.5


def test_success_reward():
    r = SuccessReward()
    assert r(_make_traj(0, True)) == 1.0
    assert r(_make_traj(0, False)) == 0.0


def test_step_penalty_reward():
    r = StepPenaltyReward(success_reward=1.0, step_penalty=0.1)
    assert r(_make_traj(0, True, 3)) == 0.7  # 1.0 - 0.3
    assert r(_make_traj(0, False, 5)) == -0.5  # 0.0 - 0.5


def test_composite_reward():
    r = CompositeReward([
        (SuccessReward(), 0.5),
        (EnvironmentReward(), 0.5),
    ])
    assert r(_make_traj(2.0, True)) == 1.5  # 0.5*1.0 + 0.5*2.0
