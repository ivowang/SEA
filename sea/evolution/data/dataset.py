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
    # Group trajectories by their initial observation (the actual task/goal),
    # NOT by task_id (which can be a fabricated per-process counter like "game_1")
    by_initial_obs: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        if traj.steps:
            key = traj.steps[0].observation.text[:200]
        else:
            key = traj.task_id
        by_initial_obs.setdefault(key, []).append(traj)

    pairs = []
    for task_key, task_trajs in by_initial_obs.items():
        task_id = task_trajs[0].task_id if task_trajs else ""
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
                # Only pair steps that share the SAME observation text.
                # Once trajectories diverge, later steps are incomparable.
                for gi, gs in enumerate(g.steps):
                    for bi, bs in enumerate(b.steps):
                        if gs.observation.text == bs.observation.text:
                            chosen = step_to_response(gs)
                            rejected = step_to_response(bs)
                            if chosen != rejected:
                                pairs.append({
                                    "prompt": gs.observation.text,
                                    "chosen": chosen,
                                    "rejected": rejected,
                                    "task_id": task_id,
                                })
                            break  # only first match per good step

    logger.info("Created %d preference pairs from %d task groups", len(pairs), len(by_initial_obs))
    return pairs


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted cumulative returns G_t for each step.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    """
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def trajectories_to_reinforce_data(
    trajectories: list[Trajectory],
    system_prompt: str = "",
    gamma: float = 0.99,
) -> list[dict[str, Any]]:
    """Convert trajectories to REINFORCE training records.

    For each step in each trajectory, produces:
    - context_messages: conversation history up to the observation at this step
    - action_text: the agent's response (raw ReAct format)
    - advantage: normalized return (G_t - mean) / std

    Returns:
        List of dicts with context_messages, action_text, advantage, return keys.
    """
    all_records: list[dict[str, Any]] = []
    all_returns: list[float] = []

    # First pass: compute returns and build records
    for traj in trajectories:
        if not traj.steps:
            continue

        step_rewards = [s.reward for s in traj.steps]
        returns = compute_returns(step_rewards, gamma)
        all_returns.extend(returns)

        # Build system message
        full_system = system_prompt
        if full_system and "Thought" not in full_system and "Action" not in full_system:
            full_system = f"{full_system}\n\n{REACT_INSTRUCTIONS}"
        elif not full_system:
            full_system = REACT_INSTRUCTIONS

        messages: list[dict[str, str]] = [{"role": "system", "content": full_system}]

        for step, G_t in zip(traj.steps, returns):
            # User message: observation
            messages.append({"role": "user", "content": step.observation.text})

            # Action text (raw ReAct format)
            raw = step.action.metadata.get("raw_response", "")
            if raw:
                action_text = raw
            else:
                action_text = step.action.text
                thought = step.action.metadata.get("thought", "")
                if thought:
                    action_text = f"Thought: {thought}\nAction: {action_text}"

            all_records.append({
                "context_messages": list(messages),  # snapshot
                "action_text": action_text,
                "return": G_t,
                "task_id": traj.task_id,
            })

            # Add assistant turn to history for next step's context
            messages.append({"role": "assistant", "content": action_text})

            # Add environment response for next step (if available)
            if step.next_observation is not None:
                messages.append({"role": "user", "content": step.next_observation.text})

    # Second pass: normalize returns to advantages
    if all_returns:
        mean_ret = sum(all_returns) / len(all_returns)
        var = sum((r - mean_ret) ** 2 for r in all_returns) / len(all_returns)
        std_ret = var ** 0.5
        for record in all_records:
            record["advantage"] = (record["return"] - mean_ret) / max(std_ret, 1e-8)
    else:
        for record in all_records:
            record["advantage"] = 0.0

    logger.info(
        "Converted %d trajectories to %d REINFORCE records",
        len(trajectories), len(all_records),
    )
    return all_records


def to_hf_dataset(data: list[dict[str, Any]]):
    """Convert a list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(data)
