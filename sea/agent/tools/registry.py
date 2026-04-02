"""Tool registry for managing available tools."""

from __future__ import annotations

import logging
from typing import Any

from sea.agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages a set of tools the agent can use.

    Tools are registered by name and can be retrieved individually or as
    a list of OpenAI function specs for LLM tool-calling.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @classmethod
    def with_builtins(cls) -> ToolRegistry:
        """Create a registry pre-loaded with built-in tools."""
        reg = cls()
        from sea.agent.tools.builtins import CalculatorTool, JSONParserTool
        reg.register(CalculatorTool())
        reg.register(JSONParserTool())
        return reg

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning("Overwriting tool '%s'", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a named tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(output=f"Unknown tool: {tool_name}", success=False)
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e)
            return ToolResult(output=f"Tool error: {e}", success=False)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_openai_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI function specs for all tools."""
        return [t.to_openai_spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
