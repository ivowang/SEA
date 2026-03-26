"""Built-in tools: basic utilities available to all agents."""

from __future__ import annotations

import json
import math
from typing import Any

from sea.agent.tools.base import Tool, ToolResult
from sea.core.registry import TOOL_REGISTRY


@TOOL_REGISTRY.register("calculator")
class CalculatorTool(Tool):
    """Evaluates simple mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Input: a math expression string."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "", **kwargs: Any) -> ToolResult:
        safe_dict = {
            k: getattr(math, k) for k in dir(math) if not k.startswith("_")
        }
        safe_dict.update({"abs": abs, "round": round, "min": min, "max": max})
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)  # noqa: S307
            return ToolResult(output=str(result))
        except Exception as e:
            return ToolResult(output=f"Error: {e}", success=False)



@TOOL_REGISTRY.register("json_parser")
class JSONParserTool(Tool):
    """Parse and extract data from JSON strings."""

    @property
    def name(self) -> str:
        return "json_parser"

    @property
    def description(self) -> str:
        return "Parse a JSON string. Optionally extract a value by dot-notation key path."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "json_string": {"type": "string", "description": "JSON string to parse"},
                "key_path": {"type": "string", "description": "Dot-notation path (e.g. 'data.items.0')"},
            },
            "required": ["json_string"],
        }

    def execute(self, json_string: str = "", key_path: str = "", **kwargs: Any) -> ToolResult:
        try:
            data = json.loads(json_string)
            if key_path:
                for key in key_path.split("."):
                    if isinstance(data, dict):
                        data = data[key]
                    elif isinstance(data, list):
                        data = data[int(key)]
                    else:
                        return ToolResult(output=f"Cannot traverse into {type(data)}", success=False)
            return ToolResult(output=json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            return ToolResult(output=f"Error: {e}", success=False)
