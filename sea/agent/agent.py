"""SEAAgent: the composed self-evolving agent.

Brings together Brain, Memory, Planner, SkillLibrary, and ToolRegistry
into a single agent that can interact with SEAEnv environments.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.agent.brain import LLMBrain
from sea.agent.memory.base import Memory, MemoryEntry
from sea.agent.planner import Planner, PlanningContext
from sea.agent.skills.library import SkillLibrary
from sea.agent.tools.base import ToolResult
from sea.agent.tools.registry import ToolRegistry
from sea.core.base import Checkpointable, Evolvable
from sea.core.types import Action, Observation, Step, Trajectory

if TYPE_CHECKING:
    from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


class SEAAgent(Checkpointable):
    """The composed self-evolving agent.

    Each subcomponent may or may not be Evolvable.  The agent itself is
    Checkpointable so its full state can be snapshotted.
    """

    def __init__(
        self,
        brain: LLMBrain,
        memory: Memory,
        planner: Planner,
        skill_library: SkillLibrary | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_retrieval_k: int = 5,
        skill_retrieval_k: int = 3,
    ) -> None:
        self.brain = brain
        self.memory = memory
        self.planner = planner
        self.skill_library = skill_library or SkillLibrary()
        self.tool_registry = tool_registry or ToolRegistry()
        self._memory_k = memory_retrieval_k
        self._skill_k = skill_retrieval_k

    def act(self, observation: Observation, task_description: str = "", step: int = 0) -> Action:
        """Single-step action selection.

        1. Retrieve relevant memories
        2. Retrieve relevant skills
        3. Build planning context
        4. Ask planner for next action
        5. Execute tool call if needed
        """
        # Retrieve from memory and skills
        query = observation.text[:500]
        retrieved_memories = self.memory.retrieve(query, k=self._memory_k)
        retrieved_skills = (
            self.skill_library.retrieve(query, k=self._skill_k)
            if len(self.skill_library) > 0
            else []
        )

        context = PlanningContext(
            observation=observation,
            retrieved_memories=retrieved_memories,
            retrieved_skills=retrieved_skills,
            available_tools=self.tool_registry.list_tools(),
            task_description=task_description,
            step_number=step,
        )

        action = self.planner.plan(self.brain, context)

        # Handle tool calls
        if action.action_type == "tool_call" and "tool_name" in action.metadata:
            tool_name = action.metadata["tool_name"]
            try:
                raw_args = action.metadata.get("tool_args_raw", "{}")
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                tool_args = {"input": raw_args}
            result = self.tool_registry.execute(tool_name, **tool_args)
            action.metadata["tool_result"] = result.output
            action.metadata["tool_success"] = result.success

        return action

    def run_episode(
        self,
        env: SEAEnv,
        task_id: str | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        """Run a complete episode in the given environment.

        Returns:
            Trajectory with all steps, total reward, and success flag.
        """
        max_steps = max_steps or env.max_steps
        self.planner.reset()

        obs, info = env.reset(task_id=task_id)
        task_desc = info.get("task_description", "")
        actual_task_id = info.get("task_id", task_id or "")

        trajectory = Trajectory(task_id=actual_task_id)
        trajectory.metadata["start_time"] = time.time()

        for step_num in range(max_steps):
            action = self.act(obs, task_description=task_desc, step=step_num)

            # Check if agent wants to finish
            if action.action_type == "finish" or action.metadata.get("finished"):
                # Submit final action to env
                obs_next, reward, terminated, truncated, step_info = env.step(action)
                trajectory.steps.append(
                    Step(observation=obs, action=action, reward=reward, done=True, info=step_info)
                )
                # Store experience in memory
                self.memory.add(MemoryEntry(
                    content=f"Step {step_num}: {action.text} -> reward={reward}",
                    memory_type="episodic",
                    metadata={"task_id": actual_task_id, "step": step_num},
                ))
                break

            obs_next, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            trajectory.steps.append(
                Step(observation=obs, action=action, reward=reward, done=done, info=step_info)
            )

            # Store experience in memory
            self.memory.add(MemoryEntry(
                content=f"Step {step_num}: {action.text} -> {obs_next.text[:200]}",
                memory_type="episodic",
                metadata={"task_id": actual_task_id, "step": step_num, "reward": reward},
            ))

            if done:
                break

            obs = obs_next

        trajectory.compute_total_reward()
        trajectory.success = trajectory.total_reward > 0 or any(
            s.info.get("success", False) for s in trajectory.steps
        )
        trajectory.metadata["end_time"] = time.time()
        trajectory.metadata["num_steps"] = len(trajectory.steps)

        logger.info(
            "Episode complete: task=%s, steps=%d, reward=%.2f, success=%s",
            actual_task_id,
            len(trajectory.steps),
            trajectory.total_reward,
            trajectory.success,
        )
        return trajectory

    def evolvable_components(self) -> dict[str, Evolvable]:
        """Return all components that implement Evolvable, keyed by name."""
        result: dict[str, Evolvable] = {}
        candidates = [
            ("brain", self.brain),
            ("memory", self.memory),
            ("skill_library", self.skill_library),
        ]
        for name, component in candidates:
            if isinstance(component, Evolvable):
                result[name] = component
        return result

    # -- Checkpointable --

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.brain.save_checkpoint(path / "brain")
        self.memory.save_checkpoint(path / "memory")
        if self.skill_library:
            self.skill_library.save_checkpoint(path / "skills")
        logger.info("Agent checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        if (path / "brain").exists():
            self.brain.load_checkpoint(path / "brain")
        if (path / "memory").exists():
            self.memory.load_checkpoint(path / "memory")
        if (path / "skills").exists() and self.skill_library:
            self.skill_library.load_checkpoint(path / "skills")
        logger.info("Agent checkpoint loaded from %s", path)

    def state_dict(self) -> dict[str, Any]:
        return {
            "brain": self.brain.state_dict(),
            "memory_size": self.memory.size(),
            "num_skills": len(self.skill_library) if self.skill_library else 0,
            "num_tools": len(self.tool_registry),
            "evolvable_components": list(self.evolvable_components().keys()),
        }
