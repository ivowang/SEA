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
        """Evaluate math expression using AST-based safe evaluation."""
        import ast
        import operator

        # Safe operators for mathematical expressions
        _ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Safe math functions
        _funcs = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
        }

        def _safe_eval(node):
            if isinstance(node, ast.Expression):
                return _safe_eval(node.body)
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value}")
            elif isinstance(node, ast.BinOp):
                op = _ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(_safe_eval(node.left), _safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = _ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(_safe_eval(node.operand))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _funcs:
                    args = [_safe_eval(a) for a in node.args]
                    return _funcs[node.func.id](*args)
                raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
            elif isinstance(node, ast.Name):
                if node.id in _funcs:
                    return _funcs[node.id]
                raise ValueError(f"Unknown name: {node.id}")
            else:
                raise ValueError(f"Unsupported expression: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
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
