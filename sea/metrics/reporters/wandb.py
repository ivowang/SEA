"""Weights & Biases reporter."""

from __future__ import annotations

from typing import Any

from sea.core.registry import REPORTER_REGISTRY
from sea.metrics.tracker import MetricsReporter


@REPORTER_REGISTRY.register("wandb")
class WandBReporter(MetricsReporter):
    """Logs metrics to Weights & Biases."""

    def __init__(
        self,
        project: str = "sea",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config or {})
        except ImportError:
            raise ImportError("wandb required. Install with: pip install wandb")

    def report(self, metrics: dict[str, float], step: int) -> None:
        log_dict = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        log_dict["step"] = step
        self._wandb.log(log_dict)

    def close(self) -> None:
        self._wandb.finish()
