"""Abstract base for evolution methods (Evolvers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.core.base import Checkpointable, Evolvable
from sea.core.types import Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.metrics.tracker import MetricsTracker


class Evolver(Checkpointable):
    """Base class for all evolution methods.

    An Evolver takes an agent, an evolution target, and collected
    trajectories, and modifies the target's state to improve performance.

    The separation of Evolver (how to evolve) and Evolvable (what to evolve)
    allows free combination of methods and targets.
    """

    @abstractmethod
    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        **kwargs,
    ) -> None:
        """Execute one evolution step.

        Modifies *target* in-place by calling target.set_evolvable_state().

        Args:
            agent: The agent (for access to brain, memory, etc.)
            target: The specific component to evolve.
            trajectories: Recently collected trajectories.
            metrics: Metrics tracker for logging.
            **kwargs: Additional context (e.g., envs for RL evolvers).
        """
        ...

    @abstractmethod
    def requires_trajectories(self) -> bool:
        """Whether this evolver needs pre-collected trajectories.

        If False, the evolver generates its own data (e.g., online RL).
        """
        ...

    # -- Default Checkpointable (evolvers typically have little state) --

    def save_checkpoint(self, path: Path) -> None:
        pass

    def load_checkpoint(self, path: Path) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}
