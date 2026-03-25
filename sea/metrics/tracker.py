"""MetricsTracker: central metrics collection and forwarding."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sea.metrics.evaluator import EvalResults

logger = logging.getLogger(__name__)


class MetricsReporter:
    """Base class for metrics output backends."""

    def report(self, metrics: dict[str, float], step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MetricsTracker:
    """Central metrics collection point.

    Stores time-series of scalar metrics, keyed by name.
    Forwards to registered reporters (console, tensorboard, wandb).
    """

    def __init__(self, reporters: list[MetricsReporter] | None = None) -> None:
        self.reporters = reporters or []
        self.global_step: int = 0
        self._history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of metrics at the current or specified step."""
        step = step if step is not None else self.global_step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._history[key].append((step, value))
        for reporter in self.reporters:
            try:
                reporter.report(metrics, step)
            except Exception as e:
                logger.warning("Reporter %s failed: %s", type(reporter).__name__, e)

    def log_eval(self, eval_results: EvalResults, step: int | None = None) -> None:
        """Log evaluation results."""
        step = step if step is not None else self.global_step
        metrics = {
            "eval/success_rate": eval_results.success_rate,
            "eval/avg_reward": eval_results.avg_reward,
            "eval/avg_steps": eval_results.avg_steps,
            "eval/num_episodes": float(eval_results.num_episodes),
        }
        for env_name, rate in eval_results.per_env.items():
            metrics[f"eval/{env_name}_success_rate"] = rate
        self.log(metrics, step)

    def get_history(self, key: str) -> list[tuple[int, float]]:
        """Return the full time-series for a metric."""
        return self._history.get(key, [])

    def latest(self, key: str) -> float | None:
        """Return the most recent value for a metric."""
        history = self._history.get(key, [])
        return history[-1][1] if history else None

    def summary(self) -> dict[str, float]:
        """Return the latest value for all metrics."""
        return {k: v[-1][1] for k, v in self._history.items() if v}

    def close(self) -> None:
        """Close all reporters."""
        for reporter in self.reporters:
            reporter.close()
