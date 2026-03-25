"""Tests for core data types."""

from sea.core.types import Action, Observation, Step, Trajectory


def test_observation_creation():
    obs = Observation(text="hello", structured={"key": "value"})
    assert obs.text == "hello"
    assert obs.structured["key"] == "value"
    assert obs.available_actions is None


def test_action_creation():
    action = Action(text="go north", action_type="text")
    assert action.text == "go north"
    assert action.action_type == "text"


def test_trajectory_length():
    traj = Trajectory(steps=[
        Step(observation=Observation(text="a"), action=Action(text="b"), reward=1.0),
        Step(observation=Observation(text="c"), action=Action(text="d"), reward=2.0),
    ])
    assert len(traj) == 2


def test_trajectory_compute_reward():
    traj = Trajectory(steps=[
        Step(observation=Observation(text="a"), action=Action(text="b"), reward=1.0),
        Step(observation=Observation(text="c"), action=Action(text="d"), reward=2.0),
    ])
    total = traj.compute_total_reward()
    assert total == 3.0
    assert traj.total_reward == 3.0
