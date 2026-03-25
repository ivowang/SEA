#!/usr/bin/env python3
"""Main entry point: load config → build components → run evolution loop."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sea.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEA: Run evolution pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dot notation (e.g., evolution.num_iterations=50)",
    )
    return parser.parse_args()


def build_from_config(cfg):
    """Build all components from a config dict."""
    from omegaconf import OmegaConf

    # Import registries to trigger @register decorators
    import sea.env.benchmarks.textcraft  # noqa: F401
    import sea.env.benchmarks.alfworld  # noqa: F401
    import sea.env.benchmarks.webshop  # noqa: F401
    import sea.evolution.methods.sft  # noqa: F401
    import sea.evolution.methods.rl  # noqa: F401
    import sea.evolution.methods.icl  # noqa: F401
    import sea.evolution.methods.prompt_evolver  # noqa: F401
    import sea.agent.tools.builtins  # noqa: F401

    from sea.core.registry import ENV_REGISTRY, EVOLVER_REGISTRY, LLM_BACKEND_REGISTRY, REPORTER_REGISTRY
    from sea.agent.agent import SEAAgent
    from sea.agent.brain import LLMBrain
    from sea.agent.memory.working import WorkingMemory
    from sea.agent.memory.semantic import SemanticMemory
    from sea.agent.memory.episodic import EpisodicMemory
    from sea.agent.planner import ReActPlanner
    from sea.agent.skills.library import SkillLibrary
    from sea.agent.tools.registry import ToolRegistry
    from sea.agent.tools.builtins import CalculatorTool, FinishTool
    from sea.evolution.pipeline import EvolutionConfig, EvolutionPipeline
    from sea.evolution.targets.lm_params import LoRATarget
    from sea.evolution.targets.prompt import PromptTarget
    from sea.metrics.tracker import MetricsTracker
    from sea.metrics.evaluator import Evaluator
    from sea.metrics.reporters.console import ConsoleReporter

    logger = logging.getLogger(__name__)

    # 1. Build LLM backend
    agent_cfg = cfg.get("agent", {})
    brain_cfg = agent_cfg.get("brain", {})
    backend_type = brain_cfg.get("backend", "vllm")
    backend_kwargs = {k: v for k, v in brain_cfg.items() if k not in ("backend", "system_prompt")}
    backend = LLM_BACKEND_REGISTRY.build(backend_type, **backend_kwargs)

    # 2. Build agent components
    system_prompt = brain_cfg.get("system_prompt", "")
    brain = LLMBrain(backend=backend, system_prompt=system_prompt)

    memory_type = agent_cfg.get("memory", "working")
    if memory_type == "semantic":
        memory = SemanticMemory()
    elif memory_type == "episodic":
        memory = EpisodicMemory()
    else:
        memory = WorkingMemory()

    planner = ReActPlanner()
    skill_library = SkillLibrary()

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(FinishTool())

    agent = SEAAgent(
        brain=brain,
        memory=memory,
        planner=planner,
        skill_library=skill_library,
        tool_registry=tool_registry,
    )

    # 3. Build environments
    env_cfg = cfg.get("env", {})
    if isinstance(env_cfg, dict):
        env_name = env_cfg.get("name", "textcraft")
        env_kwargs = {k: v for k, v in env_cfg.items() if k != "name"}
        envs = [ENV_REGISTRY.build(env_name, **env_kwargs)]
    elif isinstance(env_cfg, list):
        envs = [ENV_REGISTRY.build(e["name"], **{k: v for k, v in e.items() if k != "name"}) for e in env_cfg]
    else:
        envs = [ENV_REGISTRY.build("textcraft")]

    # 4. Build evolvers
    evo_cfg = cfg.get("evolution", {})
    evolver_cfgs = evo_cfg.get("evolvers", [])
    evolvers = []
    for ev_cfg in evolver_cfgs:
        method = ev_cfg.get("method", "sft")
        target = ev_cfg.get("target", "brain")
        ev_kwargs = {k: v for k, v in ev_cfg.items() if k not in ("method", "target")}
        if "model_name" not in ev_kwargs:
            ev_kwargs["model_name"] = brain_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
        evolver = EVOLVER_REGISTRY.build(method, **ev_kwargs)
        evolvers.append((evolver, target))

    # 5. Build metrics
    reporters_cfg = cfg.get("metrics", {}).get("reporters", ["console"])
    reporters = []
    for r in reporters_cfg:
        if isinstance(r, str):
            if r == "console":
                reporters.append(ConsoleReporter())
            elif r == "tensorboard":
                from sea.metrics.reporters.tensorboard import TensorBoardReporter
                tb_dir = cfg.get("metrics", {}).get("tensorboard_dir", "runs/sea")
                reporters.append(TensorBoardReporter(log_dir=tb_dir))
            elif r == "wandb":
                from sea.metrics.reporters.wandb import WandBReporter
                reporters.append(WandBReporter())
    metrics = MetricsTracker(reporters=reporters)

    # 6. Build evaluator
    eval_cfg = cfg.get("evaluation", {})
    evaluator = Evaluator(
        num_episodes_per_env=eval_cfg.get("num_episodes", 20),
        eval_temperature=eval_cfg.get("temperature", 0.0),
    )

    # 7. Build pipeline
    pipeline_cfg = evo_cfg.get("pipeline", {})
    pipeline_config = EvolutionConfig(
        num_iterations=pipeline_cfg.get("num_iterations", 100),
        trajectories_per_iteration=pipeline_cfg.get("traj_per_iter", 64),
        eval_every=pipeline_cfg.get("eval_every", 10),
        checkpoint_every=pipeline_cfg.get("checkpoint_every", 10),
        checkpoint_dir=pipeline_cfg.get("checkpoint_dir", "outputs/checkpoints"),
    )

    pipeline = EvolutionPipeline(
        agent=agent,
        envs=envs,
        evolvers=evolvers,
        evaluator=evaluator,
        metrics=metrics,
        config=pipeline_config,
    )

    return pipeline


def main():
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Loading config from %s", args.config)
    from sea.utils.config import load_config
    cfg = load_config(args.config, overrides=args.overrides)

    logger.info("Building components...")
    pipeline = build_from_config(cfg)

    logger.info("Starting evolution pipeline...")
    pipeline.run()

    logger.info("Done.")


if __name__ == "__main__":
    main()
