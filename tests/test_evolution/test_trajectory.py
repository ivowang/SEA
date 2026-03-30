"""Tests for trajectory buffer and data conversion."""

from sea.core.types import Action, Observation, Step, Trajectory
from sea.evolution.data.trajectory import TrajectoryBuffer
from sea.evolution.data.dataset import trajectories_to_sft_data, trajectories_to_preference_pairs


def _make_traj(task_id: str, reward: float, success: bool) -> Trajectory:
    return Trajectory(
        steps=[
            Step(observation=Observation(text="start"), action=Action(text="act"), reward=reward),
        ],
        task_id=task_id,
        total_reward=reward,
        success=success,
    )


def test_buffer_add_and_sample():
    buf = TrajectoryBuffer(max_size=100)
    for i in range(10):
        buf.add(_make_traj(f"t{i}", float(i), i > 5))
    assert len(buf) == 10

    sample = buf.sample(3)
    assert len(sample) == 3


def test_buffer_successful():
    buf = TrajectoryBuffer()
    buf.add(_make_traj("t1", 1.0, True))
    buf.add(_make_traj("t2", 0.0, False))
    buf.add(_make_traj("t3", 0.5, True))

    good = buf.successful()
    assert len(good) == 2


def test_buffer_stats():
    buf = TrajectoryBuffer()
    buf.add(_make_traj("t1", 1.0, True))
    buf.add(_make_traj("t2", 0.0, False))

    stats = buf.stats()
    assert stats["size"] == 2
    assert stats["success_rate"] == 0.5
    assert stats["avg_reward"] == 0.5


def test_trajectories_to_sft_data():
    trajs = [_make_traj("t1", 1.0, True)]
    data = trajectories_to_sft_data(trajs, system_prompt="You are helpful")
    assert len(data) == 1
    assert data[0]["messages"][0]["role"] == "system"


def test_trajectories_to_preference_pairs():
    good = Trajectory(
        steps=[Step(observation=Observation(text="start"), action=Action(text="good action"), reward=1.0)],
        task_id="t1", total_reward=1.0, success=True,
    )
    bad = Trajectory(
        steps=[Step(observation=Observation(text="start"), action=Action(text="bad action"), reward=0.0)],
        task_id="t1", total_reward=0.0, success=False,
    )
    pairs = trajectories_to_preference_pairs([good, bad])
    assert len(pairs) >= 1
    assert pairs[0]["task_id"] == "t1"
    assert "good" in pairs[0]["chosen"]
    assert "bad" in pairs[0]["rejected"]
