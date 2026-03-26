"""SEAEnv: abstract environment interface.

Follows Gymnasium conventions but uses text-centric Observation/Action
types instead of numpy arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sea.core.types import Action, Observation


class SEAEnv(ABC):
    """Base environment protocol for SEA.

    Mirrors gymnasium.Env but with text-centric types.

    Subclasses must implement:
    - reset()
    - step()
    - get_task_ids()
    - name (property)
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            task_id: Optional task identifier. If None, a random task is selected.

        Returns:
            (observation, info) tuple.
        """
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute an action in the environment.

        Args:
            action: The agent's action.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        ...

    @abstractmethod
    def get_task_ids(self) -> list[str]:
        """Return list of available task identifiers."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name."""
        ...

    @property
    def max_steps(self) -> int:
        """Maximum steps per episode. Override in subclasses."""
        return 50

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def render(self) -> str | None:
        """Return a text rendering of the current state. Optional."""
        return None
