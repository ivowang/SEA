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

Per-game selection: swap gamefiles + num_games on the batch env, then reset.
This uses ALFWorld's own observation cleaning (unlike raw TextWorld).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from sea.core.registry import ENV_REGISTRY
from sea.core.types import Action, Observation
from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)

# Task type mapping from game file directory name prefix
_TASK_TYPE_PREFIXES = [
    ("pick_and_place_simple", "pick"),
    ("pick_clean_then_place", "clean"),
    ("pick_heat_then_place", "heat"),
    ("pick_cool_then_place", "cool"),
    ("look_at_obj_in_light", "examine"),
    ("pick_two_obj", "pick_two"),
]


@ENV_REGISTRY.register("alfworld")
class ALFWorldEnv(SEAEnv):
    """ALFWorld benchmark adapter.

    Wraps the ALFWorld TextWorld environment (batch_size=1) into SEAEnv.
    Supports per-game task_id selection via gamefiles swap.
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
        # Game indexing (populated in _ensure_env)
        self._all_gamefiles: list[str] = []
        self._game_index: dict[str, str] = {}       # task_id → game_file
        self._game_type_index: dict[str, str] = {}   # task_id → task_type

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
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "configs", "envs", "alfworld_base.yaml"
            )

        # Set ALFWORLD_DATA if not set
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

        # Build game index from gamefiles
        self._all_gamefiles = list(self._env.gamefiles)
        for gf in self._all_gamefiles:
            tid = self._game_file_to_task_id(gf)
            self._game_index[tid] = gf
            self._game_type_index[tid] = self._extract_task_type_from_path(gf)

        logger.info(
            "Indexed %d games (%d unique task types)",
            len(self._game_index),
            len(set(self._game_type_index.values())),
        )

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def max_steps(self) -> int:
        return self._max_steps_val

    @staticmethod
    def _game_file_to_task_id(game_file: str) -> str:
        """Extract a stable, human-readable task_id from game file path.

        Game file structure:
            .../json_2.1.1/train/{task_dir}/{trial_dir}/game.tw-pddl

        Uses "{task_dir}/{trial_dir}" as the task_id for uniqueness
        (multiple trials per task).
        """
        parts = Path(game_file).parts
        # game.tw-pddl is at index -1, trial at -2, task at -3
        if len(parts) >= 3:
            return f"{parts[-3]}/{parts[-2]}"
        return os.path.basename(game_file)

    @staticmethod
    def _extract_task_type_from_path(game_file: str) -> str:
        """Extract task type from game file directory name prefix.

        More reliable than observation text heuristic.
        """
        # Get the task directory name (grandparent of game.tw-pddl)
        task_dir = Path(game_file).parts[-3] if len(Path(game_file).parts) >= 3 else ""
        task_dir_lower = task_dir.lower()
        for prefix, task_type in _TASK_TYPE_PREFIXES:
            if task_dir_lower.startswith(prefix):
                return task_type
        return "unknown"

    @staticmethod
    def _extract_task_type(obs_text: str) -> str:
        """Extract task type from ALFWorld observation text (fallback)."""
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

    def get_task_ids(self) -> list[str]:
        """Return real, selectable task IDs derived from game file paths."""
        self._ensure_env()
        task_ids = list(self._game_index.keys())
        if self._task_type_filter:
            task_ids = [
                tid for tid in task_ids
                if self._game_type_index.get(tid) == self._task_type_filter
            ]
        return task_ids

    def get_task_types(self) -> list[str]:
        """ALFWorld has 6 task types."""
        return ["pick", "clean", "heat", "cool", "examine", "pick_two"]

    def reset(
        self, *, seed: int | None = None, task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset to a specific game or next in sequential pool.

        Args:
            seed: Unused (ALFWorld games are deterministic).
            task_id: If provided and valid, selects that specific game.
                     If None, cycles through the game pool sequentially.
        """
        self._ensure_env()
        self._step_count = 0
        self._game_count += 1

        if task_id is not None:
            # Validate task_id
            if task_id not in self._game_index:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. Use get_task_ids() for valid IDs."
                )
            # Per-game selection: swap gamefiles to target, then reset
            game_file = self._game_index[task_id]
            self._env.gamefiles = [game_file]
            self._env.num_games = 1
            try:
                obs, infos = self._env.reset()
            finally:
                # Always restore full gamefiles list
                self._env.gamefiles = self._all_gamefiles
                self._env.num_games = len(self._all_gamefiles)

            obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            task_type = self._game_type_index.get(task_id, self._extract_task_type(obs_text))
        else:
            # Sequential cycling (original behavior)
            # If task_type_filter is set, keep resetting until match
            max_skips = 500
            for _ in range(max_skips):
                # Restore full gamefiles for sequential cycling
                self._env.gamefiles = self._all_gamefiles
                self._env.num_games = len(self._all_gamefiles)
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

            # Derive real task_id from env state if possible
            task_id = f"game_{self._game_count}"

        # Extract admissible commands
        admissible = None
        if isinstance(infos, dict) and "admissible_commands" in infos:
            cmds = infos["admissible_commands"]
            if isinstance(cmds, list) and len(cmds) > 0:
                admissible = cmds[0] if isinstance(cmds[0], list) else cmds

        info: dict[str, Any] = {
            "task_id": task_id,
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
