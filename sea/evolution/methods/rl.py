"""RL Evolver: reinforcement learning with GRPO/DPO.

Uses TRL as the training backend. Supports:
- GRPO: Group Relative Policy Optimization (online, environment-backed reward)
- DPO: Direct Preference Optimization (offline, from preference pairs)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sea.core.base import Evolvable
from sea.core.registry import EVOLVER_REGISTRY
from sea.core.types import Action, Trajectory
from sea.evolution.base import Evolver
from sea.evolution.data.dataset import (
    to_hf_dataset,
    trajectories_to_preference_pairs,
)

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv
    from sea.metrics.tracker import MetricsTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action parser: extract actions from ReAct-format LLM completions
# ---------------------------------------------------------------------------

def parse_actions_from_completion(completion: str) -> list[str]:
    """Parse action strings from a ReAct-format LLM completion.

    Handles formats like:
        Thought: I need oak logs
        Action: get 1 oak log
        Thought: Now craft planks
        Action: craft 4 oak planks using 1 oak log

    Returns a list of action strings (without the "Action:" prefix).
    If no "Action:" lines found, treats the entire text as a single action.
    """
    actions = []
    for match in re.finditer(r"Action:\s*(.+?)(?:\n|$)", completion):
        action_text = match.group(1).strip()
        if action_text and not action_text.startswith("finish"):
            actions.append(action_text)
    if not actions:
        # Fallback: use the entire completion as a single action
        clean = completion.strip()
        if clean:
            actions.append(clean)
    return actions


@EVOLVER_REGISTRY.register("rl")
class RLEvolver(Evolver):
    """RL-based evolution using GRPO or DPO via TRL.

    For GRPO, supports environment-backed rewards: the reward function
    parses model completions into action sequences, executes them in the
    environment, and returns the actual environment reward.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        algorithm: str = "grpo",  # "grpo", "dpo"
        device: str = "cuda:1",
        learning_rate: float = 1e-5,
        num_epochs: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_completion_length: int = 512,
        num_generations: int = 4,  # GRPO group size
        kl_coeff: float = 0.1,
        output_dir: str = "outputs/rl",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        lora_config: dict[str, Any] | None = None,
        envs: list[SEAEnv] | None = None,
    ) -> None:
        self._model_name = model_name
        self._algorithm = algorithm
        self._device = device
        self._lr = learning_rate
        self._epochs = num_epochs
        self._batch_size = batch_size
        self._grad_accum = gradient_accumulation_steps
        self._max_completion_length = max_completion_length
        self._num_generations = num_generations
        self._kl_coeff = kl_coeff
        self._output_dir = Path(output_dir)
        self._torch_dtype = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._lora_config = lora_config
        self._envs = envs
        self._train_step = 0

    def requires_trajectories(self) -> bool:
        return True

    def evolve(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        envs: list[SEAEnv] | None = None,
    ) -> None:
        # Use envs from evolve() call (pipeline) or from __init__ (manual)
        active_envs = envs or self._envs
        if self._algorithm == "grpo":
            self._evolve_grpo(agent, target, trajectories, metrics, active_envs)
        elif self._algorithm == "dpo":
            self._evolve_dpo(agent, target, trajectories, metrics)
        else:
            raise ValueError(f"Unsupported algorithm: {self._algorithm}")

    # ------------------------------------------------------------------
    # GRPO with environment-backed reward
    # ------------------------------------------------------------------

    def _evolve_grpo(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
        envs: list[SEAEnv] | None = None,
    ) -> None:
        """GRPO with environment-backed reward function.

        The reward function parses model completions into actions,
        executes them in the environment, and returns real rewards.

        LIMITATION: GRPO generates a single completion per prompt and
        replays parsed actions. This is a single-turn approximation —
        the model cannot observe intermediate env states during generation.
        For faithful multi-step interactive RL, use DPO with collected
        trajectory pairs instead, or implement step-wise online rollouts.
        """
        from trl import GRPOConfig, GRPOTrainer
        from sea.llm.hf_backend import HFTrainingBackend

        # Build prompt dataset from trajectory task descriptions
        # Map prompt text → task_id for env.reset()
        prompts = []
        prompt_to_task_id: dict[str, str] = {}
        for traj in trajectories:
            task_desc = traj.metadata.get("task_description", "")
            if task_desc and task_desc not in prompt_to_task_id:
                prompts.append(task_desc)
                prompt_to_task_id[task_desc] = traj.task_id

        if not prompts:
            logger.warning("No prompts for GRPO training")
            return

        dataset = to_hf_dataset([{"prompt": p} for p in prompts])

        # Load model with LoRA
        hf = HFTrainingBackend(
            model_name=self._model_name,
            device=self._device,
            torch_dtype=self._torch_dtype,
            load_in_4bit=self._load_in_4bit,
        )
        current_adapter = None
        try:
            state = target.get_evolvable_state()
            if isinstance(state, Path) and state.exists():
                current_adapter = state
        except Exception:
            pass
        model = hf.get_trainable_model(
            adapter_path=current_adapter,
            lora_config=self._lora_config,
        )
        tokenizer = hf.get_tokenizer()

        # Build reward function
        if envs:
            reward_fn = self._make_env_reward_fn(envs[0], prompt_to_task_id)
            logger.info("GRPO using environment-backed reward function")
        else:
            reward_fn = self._make_heuristic_reward_fn()
            logger.warning("GRPO using heuristic reward (no envs provided)")

        self._train_step += 1
        run_dir = self._output_dir / f"grpo_step_{self._train_step}"

        config = GRPOConfig(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            max_completion_length=self._max_completion_length,
            num_generations=self._num_generations,
            logging_steps=10,
            save_strategy="no",
            bf16=(self._torch_dtype == "bfloat16"),
            fp16=(self._torch_dtype == "float16"),
        )

        trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )

        logger.info("Starting GRPO training: %d prompts", len(prompts))
        train_result = trainer.train()

        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": "grpo",
            "rl/num_prompts": len(prompts),
            "rl/train_step": self._train_step,
        })

        del model, trainer
        import torch
        torch.cuda.empty_cache()

    @staticmethod
    def _make_env_reward_fn(env: SEAEnv, prompt_to_task_id: dict[str, str]):
        """Create a reward function that executes actions in the environment.

        For each model completion:
        1. Parse it to extract action strings
        2. Reset the environment with the matching task_id
        3. Execute actions sequentially
        4. Return the cumulative environment reward
        """
        def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
            rewards = []
            for i, completion in enumerate(completions):
                try:
                    actions = parse_actions_from_completion(completion)
                    task_id = None
                    if prompts and i < len(prompts):
                        task_id = prompt_to_task_id.get(prompts[i])

                    obs, info = env.reset(task_id=task_id)
                    total_reward = 0.0
                    for action_text in actions:
                        obs, reward, terminated, truncated, step_info = env.step(
                            Action(text=action_text)
                        )
                        total_reward += reward
                        if terminated or truncated:
                            break
                    rewards.append(total_reward)
                except Exception as e:
                    logger.debug("Env reward failed for completion: %s", e)
                    rewards.append(0.0)
            return rewards
        return reward_fn

    @staticmethod
    def _make_heuristic_reward_fn():
        """Fallback reward function when no environment is available."""
        def reward_fn(completions: list[str], **kwargs) -> list[float]:
            return [
                1.0 if any(kw in c.lower() for kw in ["craft", "get", "success"]) else 0.0
                for c in completions
            ]
        return reward_fn

    # ------------------------------------------------------------------
    # DPO (offline, from preference pairs)
    # ------------------------------------------------------------------

    def _evolve_dpo(
        self,
        agent: SEAAgent,
        target: Evolvable,
        trajectories: list[Trajectory],
        metrics: MetricsTracker,
    ) -> None:
        """DPO: train on preference pairs from trajectories."""
        from trl import DPOConfig, DPOTrainer
        from sea.llm.hf_backend import HFTrainingBackend

        pairs = trajectories_to_preference_pairs(trajectories)
        if not pairs:
            logger.warning("No preference pairs for DPO")
            return

        dataset = to_hf_dataset(pairs)

        hf = HFTrainingBackend(
            model_name=self._model_name,
            device=self._device,
            torch_dtype=self._torch_dtype,
            load_in_4bit=self._load_in_4bit,
        )
        current_adapter = None
        try:
            state = target.get_evolvable_state()
            if isinstance(state, Path) and state.exists():
                current_adapter = state
        except Exception:
            pass
        model = hf.get_trainable_model(
            adapter_path=current_adapter,
            lora_config=self._lora_config,
        )
        tokenizer = hf.get_tokenizer()

        self._train_step += 1
        run_dir = self._output_dir / f"dpo_step_{self._train_step}"

        config = DPOConfig(
            output_dir=str(run_dir),
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._grad_accum,
            learning_rate=self._lr,
            beta=self._kl_coeff,
            logging_steps=10,
            save_strategy="no",
            bf16=(self._torch_dtype == "bfloat16"),
            fp16=(self._torch_dtype == "float16"),
        )

        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training: %d pairs", len(pairs))
        train_result = trainer.train()

        adapter_path = run_dir / "adapter"
        hf.save_adapter(model, adapter_path)
        target.set_evolvable_state(adapter_path)
        agent.brain.swap_lora(str(adapter_path))

        metrics.log({
            "rl/algorithm": "dpo",
            "rl/num_pairs": len(pairs),
            "rl/train_step": self._train_step,
        })

        del model, trainer
        import torch
        torch.cuda.empty_cache()
