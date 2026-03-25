"""Tests for tools and tool registry."""

from sea.agent.tools.builtins import CalculatorTool, FinishTool
from sea.agent.tools.registry import ToolRegistry


def test_calculator_tool():
    calc = CalculatorTool()
    result = calc.execute(expression="2 + 3 * 4")
    assert result.success
    assert result.output == "14"


def test_calculator_tool_error():
    calc = CalculatorTool()
    result = calc.execute(expression="invalid")
    assert not result.success


def test_finish_tool():
    finish = FinishTool()
    result = finish.execute(answer="42")
    assert result.success
    assert result.output == "42"
    assert result.metadata.get("finished") is True


def test_tool_registry():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(FinishTool())

    assert len(reg) == 2
    assert "calculator" in reg
    assert "finish" in reg

    result = reg.execute("calculator", expression="1+1")
    assert result.output == "2"


def test_tool_registry_unknown():
    reg = ToolRegistry()
    result = reg.execute("nonexistent")
    assert not result.success


def test_tool_openai_spec():
    calc = CalculatorTool()
    spec = calc.to_openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "calculator"
