"""Tests for trajectory → dataset conversion with next_observation."""

from sea.core.types import Action, Observation, Step, Trajectory
from sea.evolution.data.dataset import trajectories_to_sft_data


def test_sft_data_uses_next_observation():
    """Verify SFT data conversion uses next_observation, not pre-action observation."""
    traj = Trajectory(
        steps=[
            Step(
                observation=Observation(text="You are in a room."),
                action=Action(text="look around"),
                next_observation=Observation(text="You see a key on the table."),
                reward=0.0,
            ),
            Step(
                observation=Observation(text="You see a key on the table."),
                action=Action(text="take key"),
                next_observation=Observation(text="You picked up the key."),
                reward=1.0,
                done=True,
            ),
        ],
        task_id="test",
        total_reward=1.0,
        success=True,
        metadata={"task_description": "Find the key."},
    )

    data = trajectories_to_sft_data([traj], system_prompt="You are helpful.")
    assert len(data) == 1

    messages = data[0]["messages"]
    # system, user(task), assistant(action1), user(env_response), assistant(action2)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Find the key."
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "look around"
    # The env response should be next_observation, NOT pre-action observation
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "You see a key on the table."
    assert messages[4]["role"] == "assistant"
    assert messages[4]["content"] == "take key"


def test_sft_data_reads_task_description_from_metadata():
    """Verify task_description comes from trajectory metadata."""
    traj = Trajectory(
        steps=[Step(observation=Observation(text="obs"), action=Action(text="act"))],
        task_id="t1",
        metadata={"task_description": "Do the thing."},
    )
    data = trajectories_to_sft_data([traj])
    assert data[0]["messages"][0]["content"] == "Do the thing."
