"""Integration test for TextCraft adapter."""

import pytest

try:
    from textcraft.env import TextCraft
    HAS_TEXTCRAFT = True
except ImportError:
    HAS_TEXTCRAFT = False

from sea.core.types import Action


@pytest.mark.skipif(not HAS_TEXTCRAFT, reason="textcraft not installed")
def test_textcraft_adapter():
    from sea.env.benchmarks.textcraft import TextCraftEnv

    env = TextCraftEnv(max_steps_val=5)
    assert env.name == "textcraft"

    obs, info = env.reset(task_id="seed_42")
    assert "task_description" in info
    assert len(obs.text) > 0
    assert "craft" in obs.text.lower() or "goal" in obs.text.lower()

    obs2, r, term, trunc, info2 = env.step(Action(text="inventory"))
    assert isinstance(obs2.text, str)
    assert isinstance(r, float)

    env.close()
