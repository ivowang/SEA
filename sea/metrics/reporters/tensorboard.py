"""TensorBoard reporter."""

from __future__ import annotations

from typing import Any

from sea.core.registry import REPORTER_REGISTRY
from sea.metrics.tracker import MetricsReporter


@REPORTER_REGISTRY.register("tensorboard")
class TensorBoardReporter(MetricsReporter):
    """Logs metrics to TensorBoard."""

    def __init__(self, log_dir: str = "runs/sea") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            raise ImportError("tensorboard required. Install with: pip install tensorboard")

    def report(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
