#!/usr/bin/env python3
"""Memory Evolution Demo: Reflexion on Riddle Room tasks.

The agent faces rooms with hidden rules it must discover through trial and error.
After each iteration, failed trajectories produce reflections stored in memory.
In subsequent iterations, retrieved reflections help avoid repeated mistakes.

Usage:
    export SEA_API_KEY="your-api-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"   # or http://localhost:8000/v1
    export SEA_MODEL="openai/gpt-5.4-nano"               # or local model name
    python examples/memory_evolution/run.py

To use a local vLLM server instead:
    python scripts/serve_model.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
    export SEA_BASE_URL="http://localhost:8000/v1"
    export SEA_MODEL="Qwen/Qwen2.5-7B-Instruct"
    python examples/memory_evolution/run.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("memory_evolution")

# ──────────────────────────────────────────────────────────────────────
# Config from environment variables
# ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 12

# ──────────────────────────────────────────────────────────────────────
# 1. Riddle Room Environment
# ──────────────────────────────────────────────────────────────────────
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

ROOMS = {
    "red_room": {
        "description": (
            "You are in a red room. There is a light switch on the wall, "
            "a locked chest in the corner, and a door. "
            "Find the treasure and exit."
        ),
        "solution": ["turn off light", "open chest", "take treasure", "exit"],
        "hint_on_fail": {
            "open chest": "The chest has a combination lock. Maybe you need darkness to see the glowing digits.",
            "take treasure": "The chest is locked. You need to open it first.",
        },
    },
    "blue_room": {
        "description": (
            "You are in a blue room. There is a note on the table, "
            "a key hanging on a hook behind a painting, and a locked door. "
            "Find the key and escape."
        ),
        "solution": ["read note", "move painting", "take key", "unlock door"],
        "hint_on_fail": {
            "take key": "You don't see any key. Maybe look around more carefully.",
            "unlock door": "You don't have a key yet.",
        },
    },
    "green_room": {
        "description": (
            "You are in a green room. There is a rug on the floor, "
            "a bookshelf against the wall, and a window. "
            "Find the hidden exit."
        ),
        "solution": ["move rug", "open trapdoor", "climb down", "escape"],
        "hint_on_fail": {
            "open trapdoor": "You don't see a trapdoor anywhere.",
            "climb down": "There's nowhere to climb down.",
        },
    },
    "yellow_room": {
        "description": (
            "You are in a yellow room. There is a mirror on the wall, "
            "a candle on the table, and a locked cabinet. "
            "Find the secret passage."
        ),
        "solution": ["light candle", "look in mirror", "press button", "enter passage"],
        "hint_on_fail": {
            "look in mirror": "The mirror is dark, you can barely see anything.",
            "press button": "You don't see any button.",
        },
    },
    "purple_room": {
        "description": (
            "You are in a purple room. There is a piano, a vase with flowers, "
            "and a painting of a sunset. Solve the puzzle to escape."
        ),
        "solution": ["play piano", "listen to melody", "arrange flowers", "open door"],
        "hint_on_fail": {
            "arrange flowers": "You're not sure how to arrange them.",
            "open door": "The door won't budge. Something else needs to happen first.",
        },
    },
    "white_room": {
        "description": (
            "You are in a white room. There is a clock on the wall showing 3:00, "
            "a dial with numbers, and a locked vault. "
            "Open the vault to escape."
        ),
        "solution": ["read clock", "set dial to 3", "open vault", "escape"],
        "hint_on_fail": {
            "open vault": "The vault is locked. It needs a code.",
            "set dial": "The dial needs a number. Look for clues in the room.",
        },
    },
}


class RiddleRoomEnv(SEAEnv):
    """Rooms with hidden rules the agent must discover."""

    def __init__(self):
        self._room = None
        self._room_id = ""
        self._completed = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "riddle_room"

    @property
    def max_steps(self) -> int:
        return 10

    def get_task_ids(self) -> list[str]:
        return list(ROOMS.keys())

    def reset(self, *, seed=None, task_id=None):
        if seed is not None:
            random.seed(seed)
        self._room_id = task_id or random.choice(list(ROOMS.keys()))
        self._room = ROOMS[self._room_id]
        self._completed = []
        self._step_count = 0

        return Observation(text=self._room["description"]), {
            "task_id": self._room_id,
            "task_description": self._room["description"],
        }

    @staticmethod
    def _fuzzy_match(expected: str, action: str) -> bool:
        """Check if all keywords in expected appear in action."""
        keywords = expected.lower().split()
        action_lower = action.lower()
        return all(kw in action_lower for kw in keywords)

    def step(self, action):
        self._step_count += 1
        action_lower = action.text.lower()

        # Check which solution steps are matched
        solution = self._room["solution"]
        next_step_idx = len(self._completed)

        if next_step_idx < len(solution):
            expected = solution[next_step_idx]
            if self._fuzzy_match(expected, action_lower):
                self._completed.append(expected)
                all_done = len(self._completed) >= len(solution)
                reward = 0.25 + (0.5 if all_done else 0.0)
                obs = f"Success! You {expected}."
                if all_done:
                    obs += " You solved the puzzle!"
                else:
                    obs += f" Progress: {len(self._completed)}/{len(solution)} steps."
                return (
                    Observation(text=obs),
                    reward,
                    all_done,
                    self._step_count >= self.max_steps,
                    {"success": all_done},
                )

        # Wrong action — give contextual hint
        hint = "That doesn't seem to help."
        for trigger, hint_text in self._room.get("hint_on_fail", {}).items():
            if self._fuzzy_match(trigger, action_lower):
                hint = hint_text
                break

        obs = (
            f"{hint}\n"
            f"Progress: {len(self._completed)}/{len(solution)} steps done."
        )
        return (
            Observation(text=obs),
            0.0,
            False,
            self._step_count >= self.max_steps,
            {"success": False},
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Build agent
# ──────────────────────────────────────────────────────────────────────
from sea.llm.api_backend import APIBackend
from sea.agent.agent import SEAAgent
from sea.agent.brain import LLMBrain
from sea.agent.memory.episodic import EpisodicMemory
from sea.agent.planner import ReActPlanner
from sea.metrics.tracker import MetricsTracker
from sea.metrics.reporters.console import ConsoleReporter
from sea.metrics.evaluator import Evaluator
from sea.evolution.methods.icl import ICLEvolver
from sea.evolution.data.trajectory import TrajectoryCollector

logger.info("=" * 60)
logger.info("Memory Evolution Demo: Riddle Room")
logger.info("=" * 60)

backend = APIBackend(model=MODEL, base_url=BASE_URL, api_key=API_KEY)

agent = SEAAgent(
    brain=LLMBrain(
        backend=backend,
        system_prompt=(
            "You are a puzzle-solving agent in a room. "
            "You must figure out the correct sequence of actions to escape. "
            "Each room has hidden rules — pay attention to clues. "
            "Respond with: Thought: <reasoning> then Action: <your action>"
        ),
        default_max_tokens=200,
        default_temperature=0.7,
    ),
    memory=EpisodicMemory(max_size=500),
    planner=ReActPlanner(),
)

env = RiddleRoomEnv()
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()

# ICL Evolver — generates reflections on failures, stores exemplars from successes
evolver = ICLEvolver(
    max_reflections_per_step=5,
    max_exemplars=5,
    exemplar_selection="diverse",
    extract_skills=False,
)

# ──────────────────────────────────────────────────────────────────────
# 3. Baseline
# ──────────────────────────────────────────────────────────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info("Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
            baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps)

# ──────────────────────────────────────────────────────────────────────
# 4. Evolution loop
# ──────────────────────────────────────────────────────────────────────
results_history = [("baseline", baseline.success_rate, baseline.avg_reward)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (memory size: %d)",
                iteration, NUM_ITERATIONS, agent.memory.size())
    logger.info("=" * 60)

    # Collect
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    success_count = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", success_count, len(trajectories))

    # Evolve memory via ICL
    memory_target = agent.evolvable_components()["memory"]
    evolver.evolve(agent, memory_target, trajectories, metrics)
    logger.info("Memory size after evolution: %d", agent.memory.size())

    # Evaluate
    result = evaluator.evaluate(agent, [env])
    results_history.append((f"iter_{iteration}", result.success_rate, result.avg_reward))
    logger.info(
        "Iter %d: success=%.0f%%, reward=%.3f, steps=%.1f",
        iteration, result.success_rate * 100, result.avg_reward, result.avg_steps,
    )

# ──────────────────────────────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Evolution Summary")
logger.info("=" * 60)
for name, sr, ar in results_history:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f", name, sr * 100, ar)

improvement = results_history[-1][1] - results_history[0][1]
logger.info("Improvement: %+.0f%% success rate", improvement * 100)
logger.info("Final memory size: %d entries", agent.memory.size())

# Save
import json
output_dir = Path("outputs/memory_evolution")
output_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "model": MODEL,
    "num_iterations": NUM_ITERATIONS,
    "results": [{"stage": n, "success_rate": s, "avg_reward": r} for n, s, r in results_history],
    "final_memory_size": agent.memory.size(),
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
logger.info("Summary saved to %s", output_dir / "summary.json")
