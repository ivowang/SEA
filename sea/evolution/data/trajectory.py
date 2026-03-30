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
        target_per_type: dict[str, int] | None = None,
        n: int = 0,
        num_workers: int = 20,
        env_name: str = "alfworld",
        env_kwargs: dict | None = None,
        backend_kwargs: dict | None = None,
        system_prompt: str = "",
        max_tokens: int = 150,
        temperature: float = 0.0,
        task_type_filter: str | None = None,
        only_successful: bool = False,
        poll_interval: int = 15,
    ) -> list[Trajectory]:
        """Collect trajectories using long-running parallel subprocess workers.

        Workers run episodes in a loop, appending results to a shared JSONL
        file. The main process polls the file and kills workers once targets
        are met.

        For targeted collection (e.g., 100 successful 'pick' trajectories),
        set task_type_filter='pick' and only_successful=True. Each worker's
        ALFWorld env will skip non-matching task types at reset().

        Args:
            target_per_type: Dict of {task_type: count} targets.
            n: Flat target (used if target_per_type is None).
            num_workers: Number of parallel worker subprocesses.
            env_name: Registered environment name.
            env_kwargs: Environment constructor kwargs.
            backend_kwargs: APIBackend kwargs.
            system_prompt: Agent system prompt.
            max_tokens: Max generation tokens.
            temperature: LLM temperature.
            task_type_filter: If set, workers only collect this task type.
            only_successful: If True, workers only save successful episodes.
            poll_interval: Seconds between progress checks.

        Returns:
            List of Trajectory objects.
        """
        import time

        env_kwargs = env_kwargs or {}
        backend_kwargs = backend_kwargs or {}

        tmpdir = tempfile.mkdtemp(prefix="sea_collect_")
        shared_jsonl = Path(tmpdir) / "trajectories.jsonl"
        shared_jsonl.touch()

        worker_script = str(Path(__file__).parent / "parallel_worker.py")

        filter_desc = f" (type={task_type_filter}, success_only={only_successful})" if task_type_filter else ""
        logger.info("Starting %d workers%s, output: %s", num_workers, filter_desc, shared_jsonl)

        procs = []
        for i in range(num_workers):
            cmd = [
                sys.executable, worker_script,
                "--output", str(shared_jsonl),
                "--max-episodes", "1000",
                "--env", env_name,
                "--env-kwargs", _json.dumps(env_kwargs),
                "--backend-type", "api",
                "--backend-kwargs", _json.dumps(backend_kwargs),
                "--system-prompt", system_prompt,
                "--max-tokens", str(max_tokens),
                "--temperature", str(temperature),
            ]
            if task_type_filter:
                cmd.extend(["--task-type-filter", task_type_filter])
            if only_successful:
                cmd.append("--only-successful")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(proc)

        logger.info("  %d workers launched", num_workers)

        def read_jsonl() -> list[dict]:
            records = []
            try:
                for line in shared_jsonl.read_text().strip().split("\n"):
                    if line.strip():
                        records.append(_json.loads(line))
            except Exception:
                pass
            return records

        def count_by_type(records):
            counts: dict[str, int] = {}
            for r in records:
                tt = r.get("task_type", "unknown")
                counts[tt] = counts.get(tt, 0) + 1
            return counts

        try:
            while True:
                time.sleep(poll_interval)
                records = read_jsonl()
                counts = count_by_type(records)
                total = len(records)

                if target_per_type:
                    short = {tt: target - counts.get(tt, 0)
                             for tt, target in target_per_type.items()
                             if counts.get(tt, 0) < target}
                    logger.info("  Progress: %d total, by type: %s, still need: %s",
                                total, counts, short if short else "DONE")
                    if not short:
                        break
                else:
                    logger.info("  Progress: %d / %d", total, n)
                    if total >= n:
                        break

                alive = sum(1 for p in procs if p.poll() is None)
                if alive == 0:
                    logger.warning("All workers exited")
                    break
        finally:
            for proc in procs:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

        # Parse JSONL into Trajectory objects
        records = read_jsonl()
        trajectories: list[Trajectory] = []
        for d in records:
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
            trajectories.append(Trajectory(
                steps=steps,
                task_id=d.get("task_id", ""),
                task_type=d.get("task_type", ""),
                total_reward=d.get("total_reward", 0.0),
                success=d.get("success", False),
                metadata=d.get("metadata", {}),
            ))

        success_count = sum(1 for t in trajectories if t.success)
        logger.info(
            "Collection done: %d trajectories (success rate: %.1f%%)",
            len(trajectories),
            100 * success_count / max(len(trajectories), 1),
        )
        return trajectories

    def _log_summary(self, trajectories: list[Trajectory]) -> None:
        logger.info(
            "Collected %d trajectories (success rate: %.1f%%)",
            len(trajectories),
            100 * sum(1 for t in trajectories if t.success) / max(len(trajectories), 1),
        )
