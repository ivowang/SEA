"""Shared data types used across the entire SEA platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Agent-environment interaction types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """What the agent perceives from the environment."""

    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    available_actions: list[str] | None = None


@dataclass(frozen=True)
class Action:
    """What the agent does in the environment."""

    text: str
    action_type: str = "text"  # "text", "code", "tool_call"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single agent-environment interaction step."""

    observation: Observation
    action: Action
    next_observation: Observation | None = None
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A complete episode of agent-environment interaction."""

    steps: list[Step] = field(default_factory=list)
    task_id: str = ""
    total_reward: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def compute_total_reward(self) -> float:
        """Recompute total reward from individual steps."""
        self.total_reward = sum(s.reward for s in self.steps)
        return self.total_reward


# ---------------------------------------------------------------------------
# LLM generation types
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """Output from an LLM backend generation call."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Message types for LLM interaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d
