"""Tests for RL evolver utilities."""

from sea.evolution.methods.rl import parse_actions_from_completion


def test_parse_react_format():
    text = (
        "Thought: I need oak logs first\n"
        "Action: get 1 oak log\n"
        "Thought: Now craft planks\n"
        "Action: craft 4 oak planks using 1 oak log"
    )
    actions = parse_actions_from_completion(text)
    assert actions == ["get 1 oak log", "craft 4 oak planks using 1 oak log"]


def test_parse_single_action():
    text = "Action: get 1 stick"
    actions = parse_actions_from_completion(text)
    assert actions == ["get 1 stick"]


def test_parse_fallback_no_action_prefix():
    text = "get 1 oak log"
    actions = parse_actions_from_completion(text)
    assert actions == ["get 1 oak log"]


def test_parse_skips_finish():
    text = "Action: get 1 log\nAction: finish(done)"
    actions = parse_actions_from_completion(text)
    assert actions == ["get 1 log"]


def test_parse_empty():
    assert parse_actions_from_completion("") == []
    assert parse_actions_from_completion("   ") == []
