"""Shared test fixtures."""

from __future__ import annotations

import pytest

from sea.core.types import Action, Observation, Step, Trajectory
from sea.agent.memory.working import WorkingMemory
from sea.agent.tools.registry import ToolRegistry
from sea.env.base import SEAEnv


class MockEnv(SEAEnv):
    """Simple mock environment for testing."""

    def __init__(self, num_tasks: int = 5, max_steps_val: int = 10):
        self._num_tasks = num_tasks
        self._max_steps = max_steps_val
        self._step_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def get_task_ids(self) -> list[str]:
        return [f"mock_task_{i}" for i in range(self._num_tasks)]

    def reset(self, *, seed=None, task_id=None):
        self._step_count = 0
        return (
            Observation(text=f"Start of task {task_id or 'default'}"),
            {"task_id": task_id or "default", "task_description": "Solve the mock task"},
        )

    def step(self, action):
        self._step_count += 1
        if "finish" in action.text.lower() or "done" in action.text.lower():
            return Observation(text="Task complete"), 1.0, True, False, {"success": True}
        if self._step_count >= self._max_steps:
            return Observation(text="Max steps"), 0.0, False, True, {}
        return Observation(text=f"Step {self._step_count}"), 0.0, False, False, {}


@pytest.fixture
def mock_env():
    return MockEnv()


@pytest.fixture
def working_memory():
    return WorkingMemory(max_size=20)


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def sample_trajectory():
    return Trajectory(
        steps=[
            Step(
                observation=Observation(text="Start"),
                action=Action(text="look around"),
                reward=0.0,
            ),
            Step(
                observation=Observation(text="You see items"),
                action=Action(text="take item"),
                reward=0.5,
            ),
            Step(
                observation=Observation(text="Done"),
                action=Action(text="finish", action_type="finish"),
                reward=1.0,
                done=True,
            ),
        ],
        task_id="test_task",
        total_reward=1.5,
        success=True,
    )
