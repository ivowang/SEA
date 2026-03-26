"""ALFWorld environment adapter.

ALFWorld is a text-based environment for household robot tasks
(pick & place, clean, heat, cool, examine objects).

Requires:
    pip install alfworld[full]
    alfworld-download

Real API verified against alfworld source (github.com/alfworld/alfworld):
- Batch-style: reset() -> (List[str], dict), step([action]) -> (List[str], List[int], List[bool], dict)
- Config from YAML, env type from config['env']['type'] (typically 'AlfredTWEnv')
- info dict contains 'admissible_commands': List[List[str]]
- Reward: 0 or 1, done=True on task completion
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sea.core.registry import ENV_REGISTRY
from sea.core.types import Action, Observation
from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


@ENV_REGISTRY.register("alfworld")
class ALFWorldEnv(SEAEnv):
    """ALFWorld benchmark adapter.

    Wraps the ALFWorld TextWorld environment (batch_size=1) into SEAEnv.
    Returns single observations (unwrapped from lists).
    """

    def __init__(
        self,
        split: str = "eval_out_of_distribution",
        max_steps_val: int = 50,
        config_path: str | None = None,
    ) -> None:
        self._split = split
        self._max_steps_val = max_steps_val
        self._config_path = config_path
        self._env = None
        self._step_count = 0
        self._game_count = 0

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        try:
            import yaml
            import alfworld.agents.environment as environment
            import alfworld.agents.modules.generic as generic
        except ImportError as e:
            raise ImportError(
                "ALFWorld is required. Install with:\n"
                "  pip install alfworld[full]\n"
                "  alfworld-download"
            ) from e

        # Load config
        if self._config_path:
            with open(self._config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = generic.load_config()

        # Create environment
        env_type = config["env"]["type"]  # typically 'AlfredTWEnv'
        self._env = getattr(environment, env_type)(config, train_eval=self._split)
        self._env = self._env.init_env(batch_size=1)
        logger.info("ALFWorld loaded (split=%s, type=%s)", self._split, env_type)

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def max_steps(self) -> int:
        return self._max_steps_val

    def get_task_ids(self) -> list[str]:
        # ALFWorld uses a sequential game pool — task_id is an index hint
        # but the underlying TextWorld env cycles through games on reset()
        return [f"game_{i}" for i in range(134)]

    def reset(
        self, *, seed: int | None = None, task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._ensure_env()
        self._step_count = 0
        self._game_count += 1

        # ALFWorld cycles through its game pool sequentially on each reset.
        # task_id/seed cannot select a specific game — this is a limitation
        # of the TextWorld-based ALFWorld environment.
        obs, infos = self._env.reset()

        # Unwrap batch dimension (batch_size=1)
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)

        # Extract admissible commands
        admissible = None
        if isinstance(infos, dict) and "admissible_commands" in infos:
            cmds = infos["admissible_commands"]
            if isinstance(cmds, list) and len(cmds) > 0:
                admissible = cmds[0] if isinstance(cmds[0], list) else cmds

        info: dict[str, Any] = {
            "task_id": task_id or f"game_{self._game_count}",
            "task_description": obs_text,
        }

        return (
            Observation(text=obs_text, available_actions=admissible),
            info,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        self._step_count += 1

        # ALFWorld expects a list of actions (batch_size=1)
        obs, scores, dones, infos = self._env.step([action.text])

        # Unwrap batch dimension
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        reward = float(scores[0]) if isinstance(scores, (list, tuple)) else float(scores)
        done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)

        # Extract admissible commands
        admissible = None
        if isinstance(infos, dict) and "admissible_commands" in infos:
            cmds = infos["admissible_commands"]
            if isinstance(cmds, list) and len(cmds) > 0:
                admissible = cmds[0] if isinstance(cmds[0], list) else cmds

        truncated = self._step_count >= self._max_steps_val
        info: dict[str, Any] = {"step": self._step_count}
        if reward > 0:
            info["success"] = True

        # Check 'won' flag if available
        if isinstance(infos, dict) and "won" in infos:
            won = infos["won"]
            if isinstance(won, (list, tuple)):
                won = won[0]
            if won:
                info["success"] = True

        return (
            Observation(text=obs_text, available_actions=admissible),
            reward,
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
