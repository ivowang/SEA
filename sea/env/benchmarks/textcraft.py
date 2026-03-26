"""TextCraft environment adapter.

TextCraft is a text-based crafting game inspired by Minecraft.
The agent receives crafting recipes and a goal, then must execute
get/craft commands to produce the target item.

Requires: pip install textcraft

Real API verified against textcraft==0.0.3:
- from textcraft.env import TextCraft  (gymnasium.Env[str, str])
- reset(seed=N) -> (str, dict)
- step(action_str) -> (str, float, bool, bool, dict)
- Actions: "get N item", "craft N item using N ingredient, ...", "inventory"
- Reward: 0 or 1 (1 only on crafting the goal item)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

from sea.core.registry import ENV_REGISTRY
from sea.core.types import Action, Observation
from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


@ENV_REGISTRY.register("textcraft")
class TextCraftEnv(SEAEnv):
    """TextCraft benchmark adapter.

    Wraps the textcraft gymnasium environment into the SEAEnv interface.
    Each episode presents a crafting goal and a set of recipes (with distractors).
    The agent must figure out the correct sequence of get/craft actions.
    """

    def __init__(self, max_steps_val: int = 30, num_tasks: int = 100) -> None:
        self._max_steps_val = max_steps_val
        self._num_tasks = num_tasks
        self._env = None
        self._step_count = 0
        self._current_goal: str = ""

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        try:
            from textcraft.env import TextCraft
            import textcraft

            # TextCraft requires minecraft_dir pointing to its bundled data
            data_dir = os.path.join(os.path.dirname(textcraft.__file__), "data")
            self._env = TextCraft(minecraft_dir=data_dir)
            logger.info("TextCraft environment loaded (data: %s)", data_dir)
        except ImportError as e:
            raise ImportError(
                "TextCraft is required. Install with: pip install textcraft"
            ) from e

    @property
    def name(self) -> str:
        return "textcraft"

    @property
    def max_steps(self) -> int:
        return self._max_steps_val

    def get_task_ids(self) -> list[str]:
        return [f"seed_{i}" for i in range(self._num_tasks)]

    def reset(
        self, *, seed: int | None = None, task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._ensure_env()
        self._step_count = 0

        # Extract seed from task_id if provided (format: "seed_N")
        if task_id and task_id.startswith("seed_"):
            seed = int(task_id.split("_")[1])
        elif seed is None:
            seed = random.randint(0, 10000)

        obs_text, info = self._env.reset(seed=seed)

        # Extract goal from observation text
        self._current_goal = ""
        for line in obs_text.split("\n"):
            if line.strip().lower().startswith("goal:"):
                self._current_goal = line.strip()
                break

        return (
            Observation(text=obs_text),
            {
                "task_id": task_id or f"seed_{seed}",
                "task_description": self._current_goal or obs_text[:200],
                **info,
            },
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        self._step_count += 1

        obs_text, reward, terminated, truncated, info = self._env.step(action.text)

        # Apply our own truncation limit
        if self._step_count >= self._max_steps_val:
            truncated = True

        info["step"] = self._step_count
        if reward > 0:
            info["success"] = True

        return Observation(text=obs_text), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
