"""WebShop environment adapter.

WebShop is a simulated e-commerce environment where agents search for and
purchase products matching given instructions.

Requires:
    git clone https://github.com/princeton-nlp/webshop.git
    cd webshop && ./setup.sh -d small
    # Add webshop to PYTHONPATH or install

Real API verified against WebShop source (github.com/princeton-nlp/webshop):
- gym.make('WebAgentTextEnv-v0', observation_mode='text', ...)
- reset(session=None) -> (str, None)  [old gym 4-tuple convention]
- step(action_str) -> (str, float, bool, None)
- Actions: "search[query]", "click[element]"
- Reward: 0.0-1.0 continuous (computed on Buy Now click)
"""

from __future__ import annotations

import logging
import random
from typing import Any

from sea.core.registry import ENV_REGISTRY
from sea.core.types import Action, Observation
from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


@ENV_REGISTRY.register("webshop")
class WebShopEnv(SEAEnv):
    """WebShop benchmark adapter.

    Wraps the WebShop gymnasium environment (WebAgentTextEnv) into SEAEnv.
    Requires the webshop package to be importable (not available on PyPI —
    must be cloned from GitHub and set up manually).
    """

    def __init__(
        self,
        max_steps_val: int = 30,
        num_tasks: int = 500,
        observation_mode: str = "text",
        human_goals: bool = True,
    ) -> None:
        self._max_steps_val = max_steps_val
        self._num_tasks = num_tasks
        self._observation_mode = observation_mode
        self._human_goals = human_goals
        self._env = None
        self._step_count = 0
        self._current_session: str = ""

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        try:
            import gym
            # WebShop registers its envs on import
            import web_agent_site.envs  # noqa: F401

            self._env = gym.make(
                "WebAgentTextEnv-v0",
                observation_mode=self._observation_mode,
                human_goals=self._human_goals,
            )
            logger.info("WebShop environment loaded (mode=%s)", self._observation_mode)
        except ImportError as e:
            raise ImportError(
                "WebShop is required. It is not a pip package.\n"
                "Install from source:\n"
                "  git clone https://github.com/princeton-nlp/webshop.git\n"
                "  cd webshop && ./setup.sh -d small\n"
                "  # Then add to PYTHONPATH or pip install -e ."
            ) from e

    @property
    def name(self) -> str:
        return "webshop"

    @property
    def max_steps(self) -> int:
        return self._max_steps_val

    def get_task_ids(self) -> list[str]:
        return [str(i) for i in range(self._num_tasks)]

    def reset(
        self, *, seed: int | None = None, task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._ensure_env()
        self._step_count = 0
        self._current_session = task_id or str(random.randint(0, self._num_tasks - 1))

        # WebShop reset returns (obs_str, None) — old gym convention
        obs_text, _ = self._env.reset(session=self._current_session)

        # Extract instruction from observation
        instruction = ""
        if "[SEP]" in obs_text:
            parts = obs_text.split("[SEP]")
            if len(parts) >= 2:
                instruction = parts[1].strip()

        info: dict[str, Any] = {
            "task_id": self._current_session,
            "task_description": instruction or obs_text[:200],
        }

        # Get available actions if supported
        available = None
        if hasattr(self._env, "get_available_actions"):
            action_info = self._env.get_available_actions()
            clickables = action_info.get("clickables", [])
            has_search = action_info.get("has_search_bar", False)
            available = []
            if has_search:
                available.append("search[<query>]")
            available.extend([f"click[{c}]" for c in clickables])

        return Observation(text=obs_text, available_actions=available), info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        self._step_count += 1

        # WebShop step returns (obs_str, reward, done, None) — old gym 4-tuple
        obs_text, reward, done, _ = self._env.step(action.text)

        truncated = self._step_count >= self._max_steps_val
        info: dict[str, Any] = {"step": self._step_count}
        if reward > 0:
            info["reward_score"] = reward
        if done and reward > 0:
            info["success"] = True

        # Get available actions
        available = None
        if hasattr(self._env, "get_available_actions"):
            action_info = self._env.get_available_actions()
            clickables = action_info.get("clickables", [])
            has_search = action_info.get("has_search_bar", False)
            available = []
            if has_search:
                available.append("search[<query>]")
            available.extend([f"click[{c}]" for c in clickables])

        return (
            Observation(text=obs_text, available_actions=available),
            float(reward),
            done,
            truncated,
            info,
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
