"""Prompt Evolver: SCOPE/EvoPrompt-style prompt optimization.

Mutates the agent's system prompt based on execution traces,
evaluates variants, and selects the best-performing one.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Trajectory
from sea.evolution.base import Evolver
from sea.evolution.targets.prompt import PromptTarget

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


@EVOLVER_REGISTRY.register("prompt")
class PromptEvolver(Evolver):
    """Evolves the agent's system prompt through mutation and selection.

    Each evolution step:
    1. Analyze recent trajectories for patterns
    2. Generate N prompt variants via LLM-based mutation
    3. (Optionally) evaluate each variant on a small set of tasks
    4. Select the best variant and update the target
    """

    def __init__(
        self,
        num_variants: int = 3,
        mutation_temperature: float = 0.9,
    ) -> None:
        self._num_variants = num_variants
        self._mutation_temp = mutation_temperature

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
        current_prompt = target.get_evolvable_state()
        if not isinstance(current_prompt, str):
            logger.warning("PromptEvolver expects str state, got %s", type(current_prompt))
            return

        # Analyse trajectory patterns
        analysis = self._analyze_trajectories(agent, trajectories)

        # Generate prompt variants
        variants = self._generate_variants(agent, current_prompt, analysis)

        if not variants:
            logger.warning("No prompt variants generated")
            return

        # Evaluate variants: score each by asking the LLM to rank them
        # against the original prompt's performance analysis
        if len(variants) > 1:
            score_prompt = (
                f"Original prompt performance:\n{analysis}\n\n"
                "Rank these prompt variants from best to worst (respond with just the numbers):\n"
            )
            for i, v in enumerate(variants):
                score_prompt += f"\n--- Variant {i+1} ---\n{v[:200]}\n"

            score_output = agent.brain.generate(
                [{"role": "user", "content": score_prompt}],
                temperature=0.0, max_tokens=50,
            )
            # Try to parse ranking — use first mentioned number
            import re
            numbers = re.findall(r"\d+", score_output.text)
            if numbers:
                best_idx = int(numbers[0]) - 1
                if 0 <= best_idx < len(variants):
                    best = variants[best_idx]
                else:
                    best = variants[0]
            else:
                best = variants[0]
        else:
            best = variants[0]

        # Only update if the variant is actually different from current
        if best.strip() == current_prompt.strip():
            logger.info("Best variant is identical to current prompt, skipping")
            return

        # Update target
        target.set_evolvable_state(best)
        agent.brain.system_prompt = best

        if isinstance(target, PromptTarget):
            success_rate = sum(1 for t in trajectories if t.success) / max(len(trajectories), 1)
            target.record_performance(best, success_rate)

        metrics.log({
            "prompt/length": len(best),
            "prompt/num_variants": len(variants),
        })

        logger.info("Prompt evolved: %d chars -> %d chars", len(current_prompt), len(best))

    def _analyze_trajectories(self, agent: SEAAgent, trajectories: list[Trajectory]) -> str:
        """Summarize trajectory patterns for guiding prompt mutation."""
        successful = [t for t in trajectories if t.success]
        failed = [t for t in trajectories if not t.success]

        summary_parts = [
            f"Success rate: {len(successful)}/{len(trajectories)}",
        ]

        if failed:
            fail_actions = []
            for t in failed[:3]:
                if t.steps:
                    fail_actions.append(t.steps[-1].action.text[:100])
            summary_parts.append(f"Common failure actions: {fail_actions}")

        if successful:
            avg_steps = sum(len(t) for t in successful) / len(successful)
            summary_parts.append(f"Avg steps to success: {avg_steps:.1f}")

        return "\n".join(summary_parts)

    def _generate_variants(
        self, agent: SEAAgent, current_prompt: str, analysis: str
    ) -> list[str]:
        """Generate prompt variants using LLM-based mutation."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a prompt engineer. Given a current system prompt and "
                    "performance analysis, generate an improved version.\n"
                    "Focus on making the prompt more specific, actionable, and "
                    "effective based on the observed patterns.\n"
                    "Output ONLY the improved prompt text, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current prompt:\n{current_prompt}\n\n"
                    f"Performance analysis:\n{analysis}\n\n"
                    "Generate an improved prompt:"
                ),
            },
        ]

        variants = []
        for _ in range(self._num_variants):
            output = agent.brain.generate(
                messages, temperature=self._mutation_temp, max_tokens=1024
            )
            variant = output.text.strip()
            if variant and variant != current_prompt:
                variants.append(variant)

        return variants
