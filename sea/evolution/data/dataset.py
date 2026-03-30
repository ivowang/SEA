"""Convert trajectories to HuggingFace Datasets for SFT/RL training."""

from __future__ import annotations

import logging
from typing import Any

from sea.core.types import Trajectory

logger = logging.getLogger(__name__)

# ReAct format instructions — included in SFT system prompt
# to match the inference-time planner prompt
REACT_INSTRUCTIONS = (
    "For each step, think about what to do (Thought), then take an action (Action).\n"
    "Format: Thought: <reasoning>\\nAction: <action>"
)


def trajectories_to_sft_data(
    trajectories: list[Trajectory],
    system_prompt: str = "",
    include_observations: bool = True,
) -> list[dict[str, Any]]:
    """Convert successful trajectories to instruction-following format.

    Each trajectory becomes a multi-turn conversation suitable for SFT.
    The system prompt includes ReAct formatting instructions to match
    the inference-time prompt structure.

    Returns:
        List of dicts with "messages" key (OpenAI chat format).
    """
    data = []
    for traj in trajectories:
        if not traj.steps:
            continue
        messages: list[dict[str, str]] = []
        # Include ReAct instructions in system prompt for format consistency
        full_system = system_prompt
        if full_system and "Thought" not in full_system and "Action" not in full_system:
            full_system = f"{full_system}\n\n{REACT_INSTRUCTIONS}"
        elif not full_system:
            full_system = REACT_INSTRUCTIONS
        messages.append({"role": "system", "content": full_system})

        # First user message: initial observation (matches what planner sees at inference time)
        initial_obs = traj.steps[0].observation.text if traj.steps else ""
        task_desc = traj.metadata.get("task_description", "")
        if initial_obs:
            messages.append({"role": "user", "content": initial_obs})
        elif task_desc:
            messages.append({"role": "user", "content": task_desc})

        for step in traj.steps:
            # Agent's action as assistant message — preserve raw ReAct format
            raw = step.action.metadata.get("raw_response", "")
            if raw:
                action_text = raw
            else:
                action_text = step.action.text
                thought = step.action.metadata.get("thought", "")
                if thought:
                    action_text = f"Thought: {thought}\nAction: {action_text}"
            messages.append({"role": "assistant", "content": action_text})

            # Environment's response as user message (use next_observation if available)
            if include_observations and step != traj.steps[-1]:
                env_response = ""
                if step.next_observation is not None:
                    env_response = step.next_observation.text
                elif step.observation is not None:
                    env_response = step.observation.text
                if env_response:
                    messages.append({"role": "user", "content": env_response})

        data.append({"messages": messages, "task_id": traj.task_id, "reward": traj.total_reward})

    logger.info("Converted %d trajectories to %d SFT samples", len(trajectories), len(data))
    return data


def trajectories_to_preference_pairs(
    trajectories: list[Trajectory],
    reward_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert trajectories to preference pairs for DPO.

    Creates per-step pairs: at each observation, the chosen response is the
    action from the successful trajectory and the rejected response is from
    the failed trajectory. This matches the inference-time single-step
    decision paradigm.

    Returns:
        List of dicts with "prompt", "chosen", "rejected" keys.
    """
    by_task: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        by_task.setdefault(traj.task_id, []).append(traj)

    pairs = []
    for task_id, task_trajs in by_task.items():
        good = [t for t in task_trajs if t.total_reward > reward_threshold or t.success]
        bad = [t for t in task_trajs if t.total_reward <= reward_threshold and not t.success]

        def step_to_response(step) -> str:
            raw = step.action.metadata.get("raw_response", "")
            if raw:
                return raw
            thought = step.action.metadata.get("thought", "")
            if thought:
                return f"Thought: {thought}\nAction: {step.action.text}"
            return f"Action: {step.action.text}"

        for g in good:
            for b in bad:
                # Per-step pairs at matching positions
                min_len = min(len(g.steps), len(b.steps))
                for i in range(max(min_len, 1)):
                    if i < len(g.steps) and i < len(b.steps):
                        prompt = g.steps[i].observation.text
                        chosen = step_to_response(g.steps[i])
                        rejected = step_to_response(b.steps[i])
                    elif i < len(g.steps):
                        prompt = g.steps[i].observation.text
                        chosen = step_to_response(g.steps[i])
                        rejected = "Action: give up"
                    else:
                        break

                    if chosen != rejected:
                        pairs.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "task_id": task_id,
                        })

    logger.info("Created %d preference pairs from %d tasks", len(pairs), len(by_task))
    return pairs


def to_hf_dataset(data: list[dict[str, Any]]):
    """Convert a list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(data)
