"""Prompt as an evolution target.

Supports both textual prompt optimization and soft prompt tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sea.core.base import Evolvable

logger = logging.getLogger(__name__)


class PromptTarget(Evolvable[str]):
    """System/task prompt as an evolution target.

    Supports textual prompt optimization (SCOPE/EvoPrompt-style)
    where the prompt text itself is mutated and selected.
    """

    def __init__(
        self,
        prompt_text: str = "",
        mode: str = "textual",  # "textual" or "learnable"
        prompt_history: list[tuple[str, float]] | None = None,
    ) -> None:
        self.prompt_text = prompt_text
        self.mode = mode
        self._history: list[tuple[str, float]] = prompt_history or []
        self._version = 0

    def record_performance(self, prompt: str, score: float) -> None:
        """Record a prompt variant and its performance score."""
        self._history.append((prompt, score))

    def best_prompt(self) -> str:
        """Return the highest-scoring prompt from history."""
        if not self._history:
            return self.prompt_text
        return max(self._history, key=lambda x: x[1])[0]

    # -- Evolvable[str] --

    def get_evolvable_state(self) -> str:
        return self.prompt_text

    def set_evolvable_state(self, state: str) -> None:
        self.prompt_text = state
        self._version += 1
        logger.info("Prompt updated (version %d, length=%d)", self._version, len(state))

    def evolution_metadata(self) -> dict[str, Any]:
        return {
            "type": "prompt",
            "mode": self.mode,
            "length": len(self.prompt_text),
            "version": self._version,
            "history_size": len(self._history),
        }

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "prompt_text": self.prompt_text,
            "mode": self.mode,
            "version": self._version,
            "history": self._history[-50:],  # Keep last 50
        }
        (path / "prompt_target.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    def load_checkpoint(self, path: Path) -> None:
        fp = path / "prompt_target.json"
        if fp.exists():
            state = json.loads(fp.read_text())
            self.prompt_text = state["prompt_text"]
            self.mode = state.get("mode", "textual")
            self._version = state.get("version", 0)
            self._history = state.get("history", [])

    def state_dict(self) -> dict[str, Any]:
        return self.evolution_metadata()
