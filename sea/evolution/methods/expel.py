"""ExpeL Evolver: distill structured rules from trajectories into memory.

ExpeL-style evolution updates memory instead of model parameters. It extracts
generalizable rules from successful and failed trajectories and stores them as
semantic memory entries in the canonical form:

    IF <condition> THEN <action> BECAUSE <reason>
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from sea.agent.memory.base import MemoryEntry
from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


@EVOLVER_REGISTRY.register("expel")
class ExpeLEvolver(Evolver):
    """Extract structured rules from successful and failed trajectories."""

    def __init__(
        self,
        max_success_trajectories: int = 4,
        max_failure_trajectories: int = 4,
        max_steps_per_trajectory: int = 8,
        max_rules_per_outcome: int = 4,
        min_priority: float = 0.9,
        reward_threshold: float = 0.0,
        deduplicate: bool = True,
    ) -> None:
        self._max_success_trajectories = max_success_trajectories
        self._max_failure_trajectories = max_failure_trajectories
        self._max_steps_per_trajectory = max_steps_per_trajectory
        self._max_rules_per_outcome = max_rules_per_outcome
        self._min_priority = min_priority
        self._reward_threshold = reward_threshold
        self._deduplicate = deduplicate
        self._rules_added = 0

    def requires_trajectories(self) -> bool:
        return True

    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        **kwargs,
    ) -> None:
        successful = [
            t for t in trajectories
            if t.success or t.total_reward > self._reward_threshold
        ]
        failed = [
            t for t in trajectories
            if not t.success and t.total_reward <= self._reward_threshold
        ]

        extracted_rules: list[dict[str, Any]] = []
        if successful:
            extracted_rules.extend(
                self._extract_rules(
                    agent=agent,
                    trajectories=self._select_trajectories(successful, prefer_high_reward=True),
                    outcome="success",
                )
            )
        if failed:
            extracted_rules.extend(
                self._extract_rules(
                    agent=agent,
                    trajectories=self._select_trajectories(failed, prefer_high_reward=False),
                    outcome="failure",
                )
            )

        existing_keys = self._existing_rule_keys(agent, target) if self._deduplicate else set()
        new_entries: list[MemoryEntry] = []
        duplicate_rules = 0

        for rule in extracted_rules:
            entry = self._rule_to_entry(rule)
            if entry is None:
                continue
            rule_key = self._rule_key(entry.content)
            if self._deduplicate and rule_key in existing_keys:
                duplicate_rules += 1
                continue
            existing_keys.add(rule_key)
            new_entries.append(entry)

        self._store_entries(agent, target, new_entries)
        self._rules_added += len(new_entries)

        metrics.log({
            "expel/num_success_trajectories": len(successful),
            "expel/num_failed_trajectories": len(failed),
            "expel/rules_extracted": len(extracted_rules),
            "expel/rules_added": len(new_entries),
            "expel/rules_deduplicated": duplicate_rules,
            "expel/total_rules_added": self._rules_added,
            "expel/memory_size": agent.memory.size(),
        })

        logger.info(
            "ExpeL: %d rules added (%d extracted, %d duplicates skipped)",
            len(new_entries),
            len(extracted_rules),
            duplicate_rules,
        )

    def _select_trajectories(
        self,
        trajectories: list[Trajectory],
        *,
        prefer_high_reward: bool,
    ) -> list[Trajectory]:
        limit = (
            self._max_success_trajectories
            if prefer_high_reward
            else self._max_failure_trajectories
        )
        return sorted(
            trajectories,
            key=lambda t: t.total_reward,
            reverse=prefer_high_reward,
        )[:limit]

    def _extract_rules(
        self,
        agent: SEAAgent,
        trajectories: list[Trajectory],
        outcome: str,
    ) -> list[dict[str, Any]]:
        if not trajectories:
            return []

        outcome_instruction = (
            "Extract reusable rules about what worked and why. "
            "The action should describe what the agent should do."
            if outcome == "success"
            else "Extract reusable rules about what to avoid and why. "
            "The action should state the safer alternative or explicitly say to avoid the bad action."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You distill trajectories into reusable behavioral rules. "
                    "Return strict JSON only with this schema: "
                    '{"rules":[{"condition":"...","action":"...","reason":"...",'
                    '"priority":0.0,"evidence":"..."}]}. '
                    "Every rule must be generalizable and concise. "
                    "Do not quote raw reflections or chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Outcome type: {outcome}\n"
                    f"{outcome_instruction}\n"
                    f"Generate at most {self._max_rules_per_outcome} rules.\n\n"
                    "Trajectories:\n"
                    f"{self._summarize_trajectories(trajectories)}\n\n"
                    "Return JSON only."
                ),
            },
        ]
        output = agent.brain.generate(messages, temperature=0.2, max_tokens=700)
        rules = self._parse_rule_payload(output.text, outcome, trajectories)
        logger.info("ExpeL: extracted %d %s rules", len(rules), outcome)
        return rules[: self._max_rules_per_outcome]

    def _summarize_trajectories(self, trajectories: list[Trajectory]) -> str:
        chunks = []
        for index, traj in enumerate(trajectories, start=1):
            steps = []
            for step_index, step in enumerate(traj.steps[: self._max_steps_per_trajectory], start=1):
                line = (
                    f"  {step_index}. obs={step.observation.text[:120]!r} | "
                    f"action={step.action.text!r}"
                )
                if step.next_observation:
                    line += f" | next={step.next_observation.text[:120]!r}"
                line += f" | reward={step.reward:.2f}"
                steps.append(line)
            task_desc = traj.metadata.get("task_description", traj.task_id)
            chunks.append(
                f"Trajectory {index}: task={traj.task_id or 'unknown'} | "
                f"success={traj.success} | reward={traj.total_reward:.2f}\n"
                f"Goal: {task_desc}\n"
                f"Steps:\n{chr(10).join(steps)}"
            )
        return "\n\n".join(chunks)

    def _parse_rule_payload(
        self,
        text: str,
        outcome: str,
        trajectories: list[Trajectory],
    ) -> list[dict[str, Any]]:
        payload = self._extract_json_payload(text)
        rules: list[dict[str, Any]] = []

        if payload:
            try:
                data = json.loads(payload)
                for item in data.get("rules", []):
                    normalized = self._normalize_rule(item, outcome, trajectories)
                    if normalized:
                        rules.append(normalized)
            except json.JSONDecodeError:
                logger.warning("ExpeL: failed to parse JSON payload, falling back to text parsing")

        if rules:
            return rules

        for line in text.splitlines():
            normalized = self._parse_rule_line(line, outcome, trajectories)
            if normalized:
                rules.append(normalized)
        return rules

    def _extract_json_payload(self, text: str) -> str | None:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _normalize_rule(
        self,
        item: dict[str, Any],
        outcome: str,
        trajectories: list[Trajectory],
    ) -> dict[str, Any] | None:
        condition = self._clean_clause(item.get("condition", ""))
        action = self._clean_clause(item.get("action", ""))
        reason = self._clean_clause(item.get("reason", ""))
        if not condition or not action or not reason:
            return None

        try:
            priority = float(item.get("priority", self._min_priority))
        except (TypeError, ValueError):
            priority = self._min_priority
        priority = max(self._min_priority, min(priority, 1.0))

        return {
            "condition": condition,
            "action": action,
            "reason": reason,
            "priority": priority,
            "outcome": outcome,
            "evidence": str(item.get("evidence", "")).strip(),
            "task_ids": [t.task_id for t in trajectories if t.task_id],
            "rewards": [t.total_reward for t in trajectories],
        }

    def _parse_rule_line(
        self,
        line: str,
        outcome: str,
        trajectories: list[Trajectory],
    ) -> dict[str, Any] | None:
        match = re.search(
            r"IF\s+(.+?)\s+THEN\s+(.+?)\s+BECAUSE\s+(.+)",
            line,
            re.IGNORECASE,
        )
        if not match:
            return None
        return self._normalize_rule(
            {
                "condition": match.group(1),
                "action": match.group(2),
                "reason": match.group(3),
                "priority": self._min_priority,
            },
            outcome,
            trajectories,
        )

    def _clean_clause(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        return cleaned.strip(" .")

    def _rule_to_entry(self, rule: dict[str, Any]) -> MemoryEntry | None:
        condition = rule.get("condition", "")
        action = rule.get("action", "")
        reason = rule.get("reason", "")
        if not condition or not action or not reason:
            return None

        content = f"IF {condition} THEN {action} BECAUSE {reason}"
        priority = float(rule.get("priority", self._min_priority))
        entry = MemoryEntry(
            content=content,
            memory_type="semantic",
            metadata={
                "source": "expel",
                "rule_format": "if_then_because",
                "outcome": rule.get("outcome", "unknown"),
                "condition": condition,
                "action": action,
                "reason": reason,
                "priority": priority,
                "evidence": rule.get("evidence", ""),
                "task_ids": rule.get("task_ids", []),
                "rewards": rule.get("rewards", []),
            },
            score=priority,
        )
        return entry

    def _existing_rule_keys(self, agent: SEAAgent, target: Evolvable) -> set[str]:
        keys = {
            self._rule_key(entry.content)
            for entry in agent.memory.get_all()
            if entry.content.strip()
        }
        try:
            state = target.get_evolvable_state()
        except Exception:
            return keys

        if isinstance(state, list):
            for item in state:
                if isinstance(item, dict) and item.get("content"):
                    keys.add(self._rule_key(str(item["content"])))
        return keys

    def _rule_key(self, content: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", content.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _store_entries(
        self,
        agent: SEAAgent,
        target: Evolvable,
        entries: list[MemoryEntry],
    ) -> None:
        if not entries:
            return

        if hasattr(target, "add"):
            for entry in entries:
                target.add(entry)
            return

        try:
            state = target.get_evolvable_state()
        except Exception:
            for entry in entries:
                agent.memory.add(entry)
            return

        if isinstance(state, list):
            state.extend(entry.to_dict() for entry in entries)
            target.set_evolvable_state(state)
            return

        for entry in entries:
            agent.memory.add(entry)
