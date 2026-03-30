"""Wrappers for adapting external environments to SEAEnv."""

from __future__ import annotations

from typing import Any, Callable

from sea.core.types import Action, Observation
from sea.env.base import SEAEnv


class GymnasiumWrapper(SEAEnv):
    """Wraps a standard Gymnasium text environment into SEAEnv.

    Usage::

        import gymnasium as gym
        gym_env = gym.make("TextGame-v0")
        sea_env = GymnasiumWrapper(gym_env, name="text_game")
    """

    def __init__(
        self,
        gym_env,
        env_name: str = "gymnasium",
        obs_parser: Callable | None = None,
        task_ids: list[str] | None = None,
        max_steps_override: int | None = None,
    ) -> None:
        self._env = gym_env
        self._name = env_name
        self._obs_parser = obs_parser or (lambda x: Observation(text=str(x)))
        self._task_ids = task_ids or ["default"]
        self._max_steps = max_steps_override or 50

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, *, seed: int | None = None, task_id: str | None = None) -> tuple[Observation, dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        obs, info = self._env.reset(**kwargs)
        return self._obs_parser(obs), info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action.text)
        return self._obs_parser(obs), float(reward), terminated, truncated, info

    def get_task_ids(self) -> list[str]:
        return self._task_ids

    def close(self) -> None:
        self._env.close()


class FunctionEnv(SEAEnv):
    """Lightweight env defined by step/reset functions.

    Useful for quick prototyping and testing::

        env = FunctionEnv(
            name="echo",
            reset_fn=lambda: (Observation(text="Hello"), {}),
            step_fn=lambda a: (Observation(text=a.text), 1.0, True, False, {}),
            task_ids=["echo_task"],
        )
    """

    def __init__(
        self,
        name: str,
        reset_fn: Callable,
        step_fn: Callable,
        task_ids: list[str] | None = None,
        max_steps_val: int = 50,
    ) -> None:
        self._name = name
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._task_ids = task_ids or ["default"]
        self._max_steps = max_steps_val

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, *, seed: int | None = None, task_id: str | None = None) -> tuple[Observation, dict[str, Any]]:
        # Always pass all kwargs to let reset_fn decide what to use
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if task_id is not None:
            kwargs["task_id"] = task_id
        try:
            return self._reset_fn(**kwargs)
        except TypeError:
            # Fallback for simple reset_fn() that takes no args
            return self._reset_fn()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        return self._step_fn(action)

    def get_task_ids(self) -> list[str]:
        return self._task_ids
