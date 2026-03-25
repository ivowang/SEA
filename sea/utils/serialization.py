"""Serialization helpers for checkpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path | str) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def load_json(path: Path | str) -> Any:
    """Load data from JSON."""
    return json.loads(Path(path).read_text())


def save_pickle(data: Any, path: Path | str) -> None:
    """Save data as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path | str) -> Any:
    """Load data from pickle."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)  # noqa: S301
