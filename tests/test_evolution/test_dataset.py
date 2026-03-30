"""Tests for trajectory → dataset conversion."""

from sea.core.types import Action, Observation, Step, Trajectory
from sea.evolution.data.dataset import trajectories_to_sft_data, REACT_INSTRUCTIONS


def test_sft_data_uses_next_observation():
    """Verify SFT data conversion uses next_observation as env response."""
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
    # system (with ReAct instructions), user(initial obs), assistant(action1), user(env_response), assistant(action2)
    assert messages[0]["role"] == "system"
    assert "You are helpful" in messages[0]["content"]
    assert "Thought" in messages[0]["content"]  # ReAct instructions included
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "You are in a room."  # initial observation, not task_desc
    assert messages[2]["role"] == "assistant"
    # The env response should be next_observation
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "You see a key on the table."


def test_sft_data_reads_task_description_from_metadata():
    """When no initial observation, falls back to task_description."""
    traj = Trajectory(
        steps=[Step(observation=Observation(text=""), action=Action(text="act"))],
        task_id="t1",
        metadata={"task_description": "Do the thing."},
    )
    data = trajectories_to_sft_data([traj])
    # System prompt should include ReAct instructions
    assert "Thought" in data[0]["messages"][0]["content"]
    # First user message falls back to task_description when obs is empty
    assert data[0]["messages"][1]["content"] == "Do the thing."
