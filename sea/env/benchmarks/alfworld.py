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
        task_type_filter: str | None = None,
    ) -> None:
        self._split = split
        self._task_type_filter = task_type_filter
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
        except ImportError as e:
            raise ImportError(
                "ALFWorld is required. Install with:\n"
                "  pip install alfworld\n"
                "  alfworld-download"
            ) from e

        # Load config from provided path or default SEA config
        config_path = self._config_path
        if not config_path:
            import os
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "configs", "envs", "alfworld_base.yaml"
            )

        # Set ALFWORLD_DATA if not set
        import os
        if "ALFWORLD_DATA" not in os.environ:
            alfworld_cache = os.path.expanduser("~/.cache/alfworld")
            if os.path.exists(alfworld_cache):
                os.environ["ALFWORLD_DATA"] = alfworld_cache

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Resolve env vars in config paths
        for section in config.values():
            if isinstance(section, dict):
                for key, val in section.items():
                    if isinstance(val, str) and "$" in val:
                        section[key] = os.path.expandvars(val)

        # Create environment
        env_type = config["env"]["type"]  # 'AlfredTWEnv'
        env_cls = environment.get_environment(env_type)
        self._env = env_cls(config, train_eval=self._split)
        self._env = self._env.init_env(batch_size=1)
        logger.info("ALFWorld loaded (split=%s, type=%s)", self._split, env_type)

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def max_steps(self) -> int:
        return self._max_steps_val

    def get_task_ids(self) -> list[str]:
        """ALFWorld does NOT support task_id-based selection.

        The env cycles through its game pool sequentially on reset().
        Returns placeholder IDs for API compatibility, but reset(task_id=...)
        does NOT select a specific game. Use task_type_filter for targeted
        collection instead.
        """
        # Return ordinal placeholders — these are NOT selectable
        return [f"alfworld_{i}" for i in range(self._num_eval_games)]

    @property
    def _num_eval_games(self) -> int:
        self._ensure_env()
        return len(getattr(self._env, "game_files", [])) or 134

    def get_task_types(self) -> list[str]:
        """ALFWorld has 6 task types."""
        return ["pick", "clean", "heat", "cool", "examine", "pick_two"]

    @staticmethod
    def _extract_task_type(obs_text: str) -> str:
        """Extract task type from ALFWorld observation text."""
        obs_lower = obs_text.lower()
        if "put a clean" in obs_lower:
            return "clean"
        elif "put a hot" in obs_lower:
            return "heat"
        elif "put a cool" in obs_lower:
            return "cool"
        elif "look at" in obs_lower and "under" in obs_lower:
            return "examine"
        elif "find two" in obs_lower or "put them" in obs_lower:
            return "pick_two"
        elif "put" in obs_lower:
            return "pick"
        return "unknown"

    def reset(
        self, *, seed: int | None = None, task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset to next game in ALFWorld's sequential pool.

        NOTE: task_id is NOT used to select a specific game. ALFWorld
        cycles through its game pool deterministically. Use task_type_filter
        in the constructor for type-targeted collection.
        """
        self._ensure_env()
        self._step_count = 0
        self._game_count += 1

        # If task_type_filter is set, keep resetting until we get the target type.
        # ALFWorld cycles through its game pool, so we skip non-matching games.
        max_skips = 500  # safety limit
        for _ in range(max_skips):
            obs, infos = self._env.reset()
            obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            task_type = self._extract_task_type(obs_text)
            if self._task_type_filter is None or task_type == self._task_type_filter:
                break
        else:
            raise RuntimeError(
                f"Could not find task type '{self._task_type_filter}' after {max_skips} resets. "
                f"This task type may not exist in the '{self._split}' split."
            )

        # Extract admissible commands
        admissible = None
        if isinstance(infos, dict) and "admissible_commands" in infos:
            cmds = infos["admissible_commands"]
            if isinstance(cmds, list) and len(cmds) > 0:
                admissible = cmds[0] if isinstance(cmds[0], list) else cmds

        info: dict[str, Any] = {
            "task_id": task_id or f"game_{self._game_count}",
            "task_description": obs_text,
            "task_type": task_type,
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
