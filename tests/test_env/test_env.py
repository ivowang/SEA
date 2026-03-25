"""Tests for environment interface and wrappers."""

from sea.core.types import Action, Observation
from sea.env.wrapper import FunctionEnv


def test_function_env():
    env = FunctionEnv(
        name="test_env",
        reset_fn=lambda: (Observation(text="Hello"), {"task_id": "t1"}),
        step_fn=lambda a: (Observation(text=f"Response to {a.text}"), 1.0, True, False, {}),
        task_ids=["t1", "t2"],
    )

    assert env.name == "test_env"
    assert env.get_task_ids() == ["t1", "t2"]

    obs, info = env.reset()
    assert obs.text == "Hello"

    obs2, reward, terminated, truncated, info = env.step(Action(text="do something"))
    assert "do something" in obs2.text
    assert reward == 1.0
    assert terminated is True


def test_mock_env(mock_env):
    obs, info = mock_env.reset(task_id="mock_task_0")
    assert "mock_task_0" in obs.text

    obs2, reward, terminated, truncated, info = mock_env.step(Action(text="try something"))
    assert reward == 0.0
    assert not terminated

    obs3, reward, terminated, truncated, info = mock_env.step(Action(text="finish"))
    assert reward == 1.0
    assert terminated is True
