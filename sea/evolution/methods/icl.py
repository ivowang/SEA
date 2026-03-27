"""ICL Evolver: in-context learning via reflection and exemplar curation.

Implements Reflexion-style verbal RL:
1. Agent runs episode → gets reward/feedback
2. Self-reflection generates verbal analysis of what went wrong
3. Reflection is stored in memory
4. Next episode, reflection is retrieved and used as context

No parameter updates — evolves memory and prompt context.
"""

from __future__ import annotations

import logging
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


@EVOLVER_REGISTRY.register("icl")
class ICLEvolver(Evolver):
    """Evolution through in-context learning (no parameter updates).

    Evolves Memory and optionally SkillLibrary by:
    - Generating reflections on failed trajectories
    - Curating few-shot exemplars from successful trajectories
    - Extracting skills from successful action patterns
    """

    def __init__(
        self,
        max_reflections_per_step: int = 5,
        max_exemplars: int = 10,
        exemplar_selection: str = "diverse",  # "diverse", "recent", "highest_reward"
        extract_skills: bool = False,
    ) -> None:
        self._max_reflections = max_reflections_per_step
        self._max_exemplars = max_exemplars
        self._exemplar_selection = exemplar_selection
        self._extract_skills = extract_skills
        self._reflection_count = 0

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
        reflections_added = 0
        exemplars_added = 0

        # 1. Generate reflections on failed trajectories
        failed = [t for t in trajectories if not t.success]
        for traj in failed[: self._max_reflections]:
            reflection = self._generate_reflection(agent, traj)
            if reflection:
                agent.memory.add(MemoryEntry(
                    content=reflection,
                    memory_type="reflection",
                    metadata={
                        "task_id": traj.task_id,
                        "reward": traj.total_reward,
                        "num_steps": len(traj),
                    },
                ))
                reflections_added += 1

        # 2. Curate few-shot exemplars from successful trajectories
        successful = [t for t in trajectories if t.success]
        for traj in self._select_exemplars(successful):
            exemplar = self._trajectory_to_exemplar(traj)
            agent.memory.add(MemoryEntry(
                content=exemplar,
                memory_type="semantic",
                metadata={
                    "task_id": traj.task_id,
                    "reward": traj.total_reward,
                    "exemplar": True,
                },
            ))
            exemplars_added += 1

        # 3. Optionally extract skills
        skills_added = 0
        if self._extract_skills and successful:
            skills_added = self._extract_skills_from_trajectories(agent, successful)

        self._reflection_count += reflections_added

        metrics.log({
            "icl/reflections_added": reflections_added,
            "icl/exemplars_added": exemplars_added,
            "icl/skills_added": skills_added,
            "icl/total_reflections": self._reflection_count,
            "icl/memory_size": agent.memory.size(),
        })

        logger.info(
            "ICL: +%d reflections, +%d exemplars, +%d skills",
            reflections_added, exemplars_added, skills_added,
        )

    def _generate_reflection(self, agent: SEAAgent, traj: Trajectory) -> str:
        """Use the agent's brain to reflect on a failed trajectory."""
        steps_summary = "\n".join(
            f"Step {i}: Action='{s.action.text}' -> Reward={s.reward}"
            for i, s in enumerate(traj.steps[-8:])
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are analyzing a failed agent trajectory. "
                    "Provide a concise reflection on what went wrong and "
                    "what should be done differently. Be specific and actionable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {traj.task_id}\n"
                    f"Total reward: {traj.total_reward}\n"
                    f"Steps ({len(traj)}):\n{steps_summary}\n\n"
                    "What went wrong and what should be done differently?"
                ),
            },
        ]
        output = agent.brain.generate(messages, temperature=0.3, max_tokens=256)
        return output.text.strip()

    def _select_exemplars(self, successful: list[Trajectory]) -> list[Trajectory]:
        """Select exemplars based on the configured strategy."""
        if not successful:
            return []
        if self._exemplar_selection == "highest_reward":
            sorted_trajs = sorted(successful, key=lambda t: t.total_reward, reverse=True)
        elif self._exemplar_selection == "recent":
            sorted_trajs = list(reversed(successful))
        else:  # diverse
            sorted_trajs = sorted(successful, key=lambda t: t.total_reward, reverse=True)
        return sorted_trajs[: self._max_exemplars]

    def _trajectory_to_exemplar(self, traj: Trajectory) -> str:
        """Convert a successful trajectory to a textual exemplar."""
        steps = "\n".join(
            f"  Action: {s.action.text}" for s in traj.steps[:10]
        )
        return f"Example for task '{traj.task_id}' (reward={traj.total_reward:.2f}):\n{steps}"

    def _extract_skills_from_trajectories(
        self, agent: SEAAgent, trajectories: list[Trajectory]
    ) -> int:
        """Extract reusable skills from successful trajectories."""
        from sea.agent.skills.code_skill import TextSkill

        count = 0
        for traj in trajectories[:3]:  # Limit to avoid too many LLM calls
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract a reusable skill from this successful trajectory. "
                        "Provide: 1) A short skill name, 2) A description, "
                        "3) Step-by-step instructions that could be reused."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Task: {traj.task_id}\nSteps:\n"
                    + "\n".join(f"- {s.action.text}" for s in traj.steps),
                },
            ]
            output = agent.brain.generate(messages, temperature=0.3, max_tokens=300)
            text = output.text.strip()

            skill = TextSkill(
                name=f"skill_from_{traj.task_id}_{count}",
                description=text[:100],
                instructions=text,
                tags=["auto_extracted"],
            )
            agent.skill_library.add_skill(skill)
            count += 1

        return count
