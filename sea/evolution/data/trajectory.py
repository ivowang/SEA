"""Trajectory collection and buffering."""

from __future__ import annotations

import json as _json
import logging
import random
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from sea.core.types import Action, Observation, Step, Trajectory

if TYPE_CHECKING:
    from sea.agent.agent import SEAAgent
    from sea.env.base import SEAEnv

logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """Ring buffer of trajectories with filtering and sampling."""

    def __init__(self, max_size: int = 10000) -> None:
        self._buffer: deque[Trajectory] = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory) -> None:
        self._buffer.append(trajectory)

    def add_batch(self, trajectories: list[Trajectory]) -> None:
        for t in trajectories:
            self._buffer.append(t)

    def sample(
        self,
        n: int,
        filter_fn: Callable[[Trajectory], bool] | None = None,
    ) -> list[Trajectory]:
        candidates = list(self._buffer)
        if filter_fn:
            candidates = [t for t in candidates if filter_fn(t)]
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    def successful(self, threshold: float = 0.0) -> list[Trajectory]:
        return [t for t in self._buffer if t.success or t.total_reward > threshold]

    def failed(self) -> list[Trajectory]:
        return [t for t in self._buffer if not t.success and t.total_reward <= 0]

    def all(self) -> list[Trajectory]:
        return list(self._buffer)

    def by_task_type(self, task_type: str) -> list[Trajectory]:
        return [t for t in self._buffer if t.task_type == task_type]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict[str, float]:
        if not self._buffer:
            return {"size": 0, "success_rate": 0.0, "avg_reward": 0.0}
        successes = sum(1 for t in self._buffer if t.success)
        avg_reward = sum(t.total_reward for t in self._buffer) / len(self._buffer)
        return {
            "size": len(self._buffer),
            "success_rate": successes / len(self._buffer),
            "avg_reward": avg_reward,
            "avg_steps": sum(len(t) for t in self._buffer) / len(self._buffer),
        }


class TrajectoryCollector:
    """Collects trajectories from agent-environment interaction."""

    def __init__(self, buffer: TrajectoryBuffer | None = None) -> None:
        self.buffer = buffer or TrajectoryBuffer()

    def collect(
        self,
        agent: SEAAgent,
        envs: list[SEAEnv],
        n: int,
        task_ids: list[str] | None = None,
        task_type_filter: str | None = None,
    ) -> list[Trajectory]:
        """Collect n trajectories sequentially.

        For parallel collection, use collect_subprocess() instead.
        """
        if task_ids is None:
            all_tasks = []
            for env in envs:
                all_tasks.extend(env.get_task_ids())
            random.shuffle(all_tasks)
        else:
            all_tasks = list(task_ids)

        assigned = [all_tasks[i % len(all_tasks)] for i in range(n)]
        trajectories: list[Trajectory] = []

        for task_id in assigned:
            try:
                traj = agent.run_episode(envs[0], task_id=task_id)
                trajectories.append(traj)
                self.buffer.add(traj)
            except Exception as e:
                logger.error("Failed to collect trajectory: %s", e)

        self._log_summary(trajectories)
        return trajectories

    @staticmethod
    def collect_subprocess(
        n: int,
        num_workers: int = 30,
        env_name: str = "alfworld",
        env_kwargs: dict | None = None,
        backend_kwargs: dict | None = None,
        system_prompt: str = "",
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> list[Trajectory]:
        """Collect trajectories using parallel subprocess workers.

        Each worker is a separate Python process with its own env + agent.
        This is the only reliable way to parallelize ALFWorld (TextWorld
        has global state that breaks threads and fork-based multiprocessing).

        Args:
            n: Total number of trajectories to collect.
            num_workers: Number of parallel worker subprocesses.
            env_name: Registered environment name.
            env_kwargs: Environment constructor kwargs.
            backend_kwargs: APIBackend kwargs (model, base_url, api_key).
            system_prompt: Agent system prompt.
            max_tokens: Max generation tokens.
            temperature: LLM temperature.

        Returns:
            List of Trajectory objects.
        """
        env_kwargs = env_kwargs or {}
        backend_kwargs = backend_kwargs or {}

        # Distribute episodes evenly
        eps_per_worker = [n // num_workers] * num_workers
        for i in range(n % num_workers):
            eps_per_worker[i] += 1
        eps_per_worker = [e for e in eps_per_worker if e > 0]
        actual_workers = len(eps_per_worker)

        logger.info(
            "Subprocess collection: %d episodes across %d workers",
            n, actual_workers,
        )

        tmpdir = tempfile.mkdtemp(prefix="sea_collect_")
        worker_script = str(Path(__file__).parent / "parallel_worker.py")
        procs = []

        for i, ep_count in enumerate(eps_per_worker):
            output_file = f"{tmpdir}/worker_{i}.json"
            cmd = [
                sys.executable, worker_script,
                "--output", output_file,
                "--n", str(ep_count),
                "--env", env_name,
                "--env-kwargs", _json.dumps(env_kwargs),
                "--backend-type", "api",
                "--backend-kwargs", _json.dumps(backend_kwargs),
                "--system-prompt", system_prompt,
                "--max-tokens", str(max_tokens),
                "--temperature", str(temperature),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            procs.append((proc, output_file))
            logger.info("  Worker %d: PID %d, %d episodes", i, proc.pid, ep_count)

        # Wait for all
        for proc, _ in procs:
            proc.wait()

        # Collect results from JSON files
        trajectories: list[Trajectory] = []
        for i, (proc, output_file) in enumerate(procs):
            try:
                data = _json.loads(Path(output_file).read_text())
                for d in data:
                    steps = [
                        Step(
                            observation=Observation(text=s.get("observation", "")),
                            action=Action(text=s.get("action", ""),
                                          action_type=s.get("action_type", "text")),
                            next_observation=(
                                Observation(text=s["next_observation"])
                                if s.get("next_observation") else None
                            ),
                            reward=s.get("reward", 0.0),
                            done=s.get("done", False),
                            info=s.get("info", {}),
                        )
                        for s in d.get("steps", [])
                    ]
                    traj = Trajectory(
                        steps=steps,
                        task_id=d.get("task_id", ""),
                        task_type=d.get("task_type", ""),
                        total_reward=d.get("total_reward", 0.0),
                        success=d.get("success", False),
                        metadata=d.get("metadata", {}),
                    )
                    trajectories.append(traj)
            except Exception as e:
                # Check stderr for error details
                stderr = procs[i][0].stderr.read().decode()[-500:] if procs[i][0].stderr else ""
                logger.error("Worker %d failed: %s\nstderr: %s", i, e, stderr)

        success_count = sum(1 for t in trajectories if t.success)
        logger.info(
            "Collected %d/%d trajectories (success rate: %.1f%%)",
            len(trajectories), n,
            100 * success_count / max(len(trajectories), 1),
        )
        return trajectories

    def _log_summary(self, trajectories: list[Trajectory]) -> None:
        logger.info(
            "Collected %d trajectories (success rate: %.1f%%)",
            len(trajectories),
            100 * sum(1 for t in trajectories if t.success) / max(len(trajectories), 1),
        )
