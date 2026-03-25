"""Abstract base class for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """A callable tool that the agent can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description for the LLM to understand when to use it."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's parameters. Override for typed tools."""
        return {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments."""
        ...

    def to_openai_spec(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {"type": "object", "properties": {}},
            },
        }
