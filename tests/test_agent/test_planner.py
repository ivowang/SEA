"""Tests for planner action parsing."""

from sea.agent.planner import ReActPlanner


def test_parse_finish():
    planner = ReActPlanner()
    action = planner._parse_action("Thought: I'm done\nAction: finish(42)")
    assert action.action_type == "finish"
    assert action.text == "42"


def test_parse_finish_with_parens():
    planner = ReActPlanner()
    action = planner._parse_action("Action: finish(The answer is (yes))")
    assert action.action_type == "finish"
    assert "yes" in action.text


def test_parse_tool_call():
    planner = ReActPlanner()
    action = planner._parse_action('Action: tool_call(calculator, {"expression": "2+2"})')
    assert action.action_type == "tool_call"
    assert action.metadata["tool_name"] == "calculator"


def test_parse_plain_action():
    planner = ReActPlanner()
    action = planner._parse_action("Thought: Let me look\nAction: go to kitchen")
    assert action.action_type == "text"
    assert action.text == "go to kitchen"
