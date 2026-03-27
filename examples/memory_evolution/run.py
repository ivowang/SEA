#!/usr/bin/env python3
"""Memory Evolution Demo: Reflexion on Riddle Room tasks.

The agent faces rooms with hidden rules it must discover through trial
and error. After each iteration, the ICL evolver generates reflections
on failed trajectories and stores exemplars from successes. In later
iterations, retrieved reflections help avoid repeated mistakes.

Usage:
    export SEA_API_KEY="your-api-key"
    export SEA_BASE_URL="https://api.aigocode.com/v1"
    export SEA_MODEL="openai/gpt-5.4-nano"
    python examples/memory_evolution/run.py

To use a local vLLM server instead:
    export SEA_BASE_URL="http://localhost:8000/v1"
    export SEA_MODEL="Qwen/Qwen2.5-7B-Instruct"
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sea.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger("memory_evolution")

# ── Config ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("SEA_API_KEY", "")
BASE_URL = os.environ.get("SEA_BASE_URL", "https://api.aigocode.com/v1")
MODEL = os.environ.get("SEA_MODEL", "openai/gpt-5.4-nano")

NUM_ITERATIONS = 4
NUM_COLLECT = 12
NUM_EVAL = 12

# ── 1. Riddle Room Environment ────────────────────────────────────────
from sea.env.base import SEAEnv
from sea.core.types import Action, Observation

ROOMS = {
    "red_room": {
        "description": (
            "You are in a red room. There is a light switch, a locked chest, "
            "and a door. Find the treasure and exit."
        ),
        "solution": ["turn off light", "open chest", "take treasure", "exit"],
    },
    "blue_room": {
        "description": (
            "You are in a blue room. There is a note on the table, "
            "a painting on the wall, and a locked door. Escape the room."
        ),
        "solution": ["read note", "move painting", "take key", "unlock door"],
    },
    "green_room": {
        "description": (
            "You are in a green room. There is a rug on the floor, "
            "a bookshelf, and a window. Find the hidden exit."
        ),
        "solution": ["move rug", "open trapdoor", "climb down", "escape"],
    },
    "yellow_room": {
        "description": (
            "You are in a yellow room. There is a mirror, a candle, "
            "and a locked cabinet. Find the secret passage."
        ),
        "solution": ["light candle", "look mirror", "press button", "enter passage"],
    },
    "purple_room": {
        "description": (
            "You are in a purple room. There is a piano, a vase, "
            "and a painting. Solve the puzzle to escape."
        ),
        "solution": ["play piano", "listen melody", "arrange flowers", "open door"],
    },
    "white_room": {
        "description": (
            "You are in a white room. There is a clock showing 3:00, "
            "a dial with numbers, and a locked vault. Open the vault."
        ),
        "solution": ["read clock", "set dial", "open vault", "escape"],
    },
}


class RiddleRoomEnv(SEAEnv):
    """Rooms with hidden rules the agent must discover."""

    def __init__(self):
        self._room = None
        self._room_id = ""
        self._completed: list[str] = []
        self._step_count = 0

    @property
    def name(self) -> str:
        return "riddle_room"

    @property
    def max_steps(self) -> int:
        return 10

    def get_task_ids(self) -> list[str]:
        return list(ROOMS.keys())

    @staticmethod
    def _fuzzy_match(expected: str, action: str) -> bool:
        keywords = expected.lower().split()
        return all(kw in action.lower() for kw in keywords)

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

    def step(self, action):
        self._step_count += 1
        solution = self._room["solution"]
        next_idx = len(self._completed)

        if next_idx < len(solution):
            expected = solution[next_idx]
            if self._fuzzy_match(expected, action.text):
                self._completed.append(expected)
                all_done = len(self._completed) >= len(solution)
                reward = 0.25 + (0.5 if all_done else 0.0)
                obs = f"Success: {expected}."
                if all_done:
                    obs += " Puzzle solved!"
                else:
                    obs += f" ({len(self._completed)}/{len(solution)})"
                return (
                    Observation(text=obs), reward, all_done,
                    self._step_count >= self.max_steps,
                    {"success": all_done},
                )

        obs = (
            f"That doesn't work. ({len(self._completed)}/{len(solution)} done). "
            f"Think about what the room description hints at."
        )
        return (
            Observation(text=obs), 0.0, False,
            self._step_count >= self.max_steps, {"success": False},
        )


# ── 2. Build agent ────────────────────────────────────────────────────
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
            "Each room has hidden rules — pay attention to clues in the description. "
            "Respond with: Thought: <reasoning> then Action: <your action>"
        ),
        default_max_tokens=200,
        default_temperature=0.5,
    ),
    memory=EpisodicMemory(max_size=500),
    planner=ReActPlanner(),
)

env = RiddleRoomEnv()
metrics = MetricsTracker(reporters=[ConsoleReporter()])
evaluator = Evaluator(num_episodes_per_env=NUM_EVAL, eval_temperature=0.0)
collector = TrajectoryCollector()

evolver = ICLEvolver(
    max_reflections_per_step=5,
    max_exemplars=5,
    exemplar_selection="diverse",
    extract_skills=False,
)

# ── 3. Baseline (eval_mode — no memory contamination) ─────────────────
logger.info("Baseline evaluation ...")
baseline = evaluator.evaluate(agent, [env])
logger.info(
    "Baseline: success=%.0f%%, reward=%.3f, steps=%.1f",
    baseline.success_rate * 100, baseline.avg_reward, baseline.avg_steps,
)

# ── 4. Evolution loop ─────────────────────────────────────────────────
results = [("baseline", baseline.success_rate, baseline.avg_reward)]

for iteration in range(1, NUM_ITERATIONS + 1):
    logger.info("=" * 60)
    logger.info("Iteration %d / %d  (memory: %d entries)",
                iteration, NUM_ITERATIONS, agent.memory.size())
    logger.info("=" * 60)

    # Collect (writes to memory via run_episode)
    trajectories = collector.collect(agent, [env], n=NUM_COLLECT)
    n_ok = sum(1 for t in trajectories if t.success)
    logger.info("Collected: %d/%d successful", n_ok, len(trajectories))

    # Evolve memory (add reflections + exemplars)
    memory_target = agent.evolvable_components()["memory"]
    evolver.evolve(agent, memory_target, trajectories, metrics)
    logger.info("Memory after evolution: %d entries", agent.memory.size())

    # Evaluate (eval_mode — side-effect-free)
    result = evaluator.evaluate(agent, [env])
    results.append((f"iter_{iteration}", result.success_rate, result.avg_reward))
    logger.info(
        "Iter %d: success=%.0f%%, reward=%.3f, steps=%.1f",
        iteration, result.success_rate * 100, result.avg_reward, result.avg_steps,
    )

# ── 5. Summary ────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Summary")
logger.info("=" * 60)
for name, sr, ar in results:
    logger.info("  %-12s  success=%.0f%%  reward=%.3f", name, sr * 100, ar)
improvement = results[-1][1] - results[0][1]
logger.info("Improvement: %+.0f%% success rate", improvement * 100)
logger.info("Final memory: %d entries", agent.memory.size())

output_dir = Path("outputs/memory_evolution")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "summary.json").write_text(json.dumps({
    "model": MODEL,
    "results": [{"stage": n, "success_rate": s, "avg_reward": r} for n, s, r in results],
    "final_memory_size": agent.memory.size(),
}, indent=2))
logger.info("Saved to %s", output_dir / "summary.json")
