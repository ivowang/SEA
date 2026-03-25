"""Console reporter using Rich for live terminal output."""

from __future__ import annotations

from typing import Any

from sea.core.registry import REPORTER_REGISTRY
from sea.metrics.tracker import MetricsReporter


@REPORTER_REGISTRY.register("console")
class ConsoleReporter(MetricsReporter):
    """Rich-based terminal reporter showing metrics as a live table."""

    def __init__(self, print_every: int = 1) -> None:
        self._print_every = print_every
        self._call_count = 0
        try:
            from rich.console import Console
            from rich.table import Table
            self._console = Console()
            self._has_rich = True
        except ImportError:
            self._has_rich = False

    def report(self, metrics: dict[str, float], step: int) -> None:
        self._call_count += 1
        if self._call_count % self._print_every != 0:
            return

        if self._has_rich:
            from rich.table import Table

            table = Table(title=f"Step {step}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")

            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))

            self._console.print(table)
        else:
            parts = [f"[Step {step}]"]
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value}")
            print("  ".join(parts))
