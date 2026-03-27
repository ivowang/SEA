"""Planner: determines the agent's next action given context.

Includes ReActPlanner (Thought -> Action -> Observation loop) as the
default implementation.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sea.core.registry import PLANNER_REGISTRY
from sea.core.types import Action, Observation

if TYPE_CHECKING:
    from sea.agent.brain import LLMBrain
    from sea.agent.memory.base import MemoryEntry
    from sea.agent.skills.base import Skill
    from sea.agent.tools.base import Tool

logger = logging.getLogger(__name__)


@dataclass
class PlanningContext:
    """All context available to the planner for a single step."""

    observation: Observation
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memories: list[Any] = field(default_factory=list)
    retrieved_skills: list[Any] = field(default_factory=list)
    available_tools: list[Any] = field(default_factory=list)
    task_description: str = ""
    step_number: int = 0


class Planner(ABC):
    """Abstract planner interface."""

    @abstractmethod
    def plan(self, brain: LLMBrain, context: PlanningContext) -> Action:
        """Given context, decide the next action."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset planner state for a new episode."""
        ...


@PLANNER_REGISTRY.register("react")
class ReActPlanner(Planner):
    """ReAct-style planner: Thought -> Action -> Observation loop.

    The LLM generates a thought (reasoning) followed by an action.
    The action is parsed and returned.
    """

    SYSTEM_PROMPT = (
        "You are a helpful agent that solves tasks step by step.\n"
        "For each step, first think about what to do (Thought), "
        "then take an action (Action).\n\n"
        "Format your response EXACTLY as:\n"
        "Thought: <your reasoning>\n"
        "Action: <action text>\n\n"
        "If you want to use a tool, format as:\n"
        "Thought: <your reasoning>\n"
        "Action: tool_call(<tool_name>, <args as JSON>)\n\n"
        "When you are done, use:\n"
        "Action: finish(<your final answer>)"
    )

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries
        self._history: list[dict[str, str]] = []

    def reset(self) -> None:
        self._history.clear()

    def _build_messages(self, brain: LLMBrain, context: PlanningContext) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        # System prompt
        system_parts = [self.SYSTEM_PROMPT]
        if brain.system_prompt:
            system_parts.append(brain.system_prompt)
        if context.task_description:
            system_parts.append(f"\nTask: {context.task_description}")

        # Available tools
        if context.available_tools:
            tool_descs = []
            for t in context.available_tools:
                tool_descs.append(f"- {t.name}: {t.description}")
            system_parts.append("\nAvailable tools:\n" + "\n".join(tool_descs))

        # Retrieved skills
        if context.retrieved_skills:
            skill_parts = []
            for s in context.retrieved_skills:
                skill_parts.append(s.to_prompt())
            system_parts.append("\nRelevant skills:\n" + "\n---\n".join(skill_parts))

        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # Retrieved memories (limit to 3, truncate each to keep context clean)
        if context.retrieved_memories:
            mem_parts = []
            for m in context.retrieved_memories[:3]:
                text = m.content[:200]
                mem_parts.append(f"- [{m.memory_type}] {text}")
            messages.append({
                "role": "system",
                "content": "Relevant past experience:\n" + "\n".join(mem_parts),
            })

        # Conversation history
        messages.extend(self._history)

        # Current observation
        obs_text = context.observation.text
        if context.observation.available_actions:
            obs_text += "\nAvailable actions: " + ", ".join(context.observation.available_actions)
        messages.append({"role": "user", "content": obs_text})

        return messages

    def _parse_action(self, response: str) -> Action:
        """Extract action from the LLM response."""
        # Try to find Action: line
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
        else:
            action_text = response.strip()

        # Check for tool_call pattern: tool_call(name, args)
        tool_match = re.match(r"tool_call\((\w+),\s*(.+)\)$", action_text, re.DOTALL)
        if tool_match:
            return Action(
                text=action_text,
                action_type="tool_call",
                metadata={
                    "tool_name": tool_match.group(1),
                    "tool_args_raw": tool_match.group(2).rstrip(")"),
                },
            )

        # Check for finish pattern: finish(answer)
        finish_match = re.match(r"finish\((.+)\)$", action_text, re.DOTALL)
        if finish_match:
            return Action(
                text=finish_match.group(1),
                action_type="finish",
                metadata={"finished": True},
            )

        return Action(text=action_text, action_type="text")

    def plan(self, brain: LLMBrain, context: PlanningContext) -> Action:
        messages = self._build_messages(brain, context)
        output = brain.generate(messages)
        response = output.text

        # Store in history
        self._history.append({"role": "user", "content": context.observation.text})
        self._history.append({"role": "assistant", "content": response})

        action = self._parse_action(response)
        action.metadata["raw_response"] = response

        # Extract thought if present
        thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", response, re.DOTALL)
        if thought_match:
            action.metadata["thought"] = thought_match.group(1).strip()

        return action
