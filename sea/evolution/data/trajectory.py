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
    ) -> list[Trajectory]:
        """Collect n trajectories distributed across all environments.

        Builds (env, task_id) pairs from all environments and cycles through
        them. Each trajectory records which environment it came from in
        metadata["env_name"].

        For parallel collection, use collect_subprocess() instead.
        """
        if not envs:
            raise ValueError("At least one environment required")

        # Build (env, task_id) pairs
        pairs: list[tuple[SEAEnv, str]] = []
        if task_ids is not None:
            # Explicit task_ids: distribute round-robin across envs
            for i, tid in enumerate(task_ids):
                pairs.append((envs[i % len(envs)], tid))
        else:
            # Auto-discover from each env
            for env in envs:
                env_tasks = list(env.get_task_ids())
                for tid in env_tasks:
                    pairs.append((env, tid))

        if not pairs:
            raise ValueError("No (env, task_id) pairs available")

        random.shuffle(pairs)
        assigned = [pairs[i % len(pairs)] for i in range(n)]
        trajectories: list[Trajectory] = []

        for env, task_id in assigned:
            try:
                traj = agent.run_episode(env, task_id=task_id)
                traj.metadata["env_name"] = env.name
                trajectories.append(traj)
                self.buffer.add(traj)
            except Exception as e:
                logger.error("Failed on (env=%s, task=%s): %s", env.name, task_id, e)

        self._log_summary(trajectories)
        return trajectories

    @staticmethod
    def collect_parallel(
        agent_factory,
        env_factory,
        n: int,
        max_workers: int = 30,
        task_ids: list[str] | None = None,
        only_successful: bool = False,
    ) -> list[Trajectory]:
        """Collect n trajectories with high concurrency using thread pool.

        Each worker thread gets its own agent + env instance (created via
        factory functions) to avoid shared mutable state. The API backend
        is thread-safe, so all workers share the underlying HTTP connection pool.

        This is ideal for API-based collection where the bottleneck is
        network latency, not CPU/GPU.

        Args:
            agent_factory: Callable that returns a fresh SEAAgent instance.
                Each worker calls this once to get its own agent.
            env_factory: Callable that returns a fresh SEAEnv instance.
                Each worker calls this once to get its own environment.
            n: Number of trajectories to collect.
            max_workers: Maximum concurrent threads.
            task_ids: Optional list of task IDs to cycle through.
            only_successful: If True, only return successful trajectories
                (will collect more than n to meet the target).

        Returns:
            List of Trajectory objects.

        Example:
            trajectories = TrajectoryCollector.collect_parallel(
                agent_factory=lambda: SEAAgent(brain=LLMBrain(APIBackend(...)), ...),
                env_factory=lambda: TextCraftEnv(max_steps_val=15),
                n=100,
                max_workers=30,
            )
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results: list[Trajectory] = []
        results_lock = threading.Lock()
        completed = threading.Event()

        # Determine task assignments
        if task_ids:
            assignments = [task_ids[i % len(task_ids)] for i in range(n)]
        else:
            assignments = [None] * n

        # If only_successful, we may need to oversample
        target = n
        if only_successful:
            assignments = [None] * (n * 3)  # initial oversample 3x

        def _run_one(idx: int, task_id: str | None) -> Trajectory | None:
            """Run one episode in its own agent+env."""
            if completed.is_set():
                return None
            try:
                agent = agent_factory()
                env = env_factory()
                try:
                    traj = agent.run_episode(env, task_id=task_id)
                    traj.metadata["env_name"] = env.name
                    traj.metadata["worker_idx"] = idx
                    return traj
                finally:
                    env.close()
            except Exception as e:
                logger.error("Worker %d failed: %s", idx, e)
                return None

        logger.info(
            "Starting parallel collection: target=%d, workers=%d, only_successful=%s",
            target, min(max_workers, len(assignments)), only_successful,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, i, tid): i
                for i, tid in enumerate(assignments)
            }

            for future in as_completed(futures):
                traj = future.result()
                if traj is None:
                    continue

                if only_successful and not traj.success:
                    continue

                with results_lock:
                    results.append(traj)
                    count = len(results)

                if count % 10 == 0 or count == target:
                    logger.info("  Progress: %d/%d trajectories collected", count, target)

                if count >= target:
                    completed.set()
                    break

        success_count = sum(1 for t in results if t.success)
        logger.info(
            "Parallel collection done: %d trajectories (%.1f%% success)",
            len(results), 100 * success_count / max(len(results), 1),
        )
        return results[:target]

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
            stderr_file = Path(tmpdir) / f"worker_{i}.stderr"
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(stderr_file, "w"))
            procs.append(proc)

        logger.info("  %d workers launched", num_workers)

        def read_jsonl() -> list[dict]:
            records = []
            try:
                text = shared_jsonl.read_text()
            except Exception:
                return records
            for line in text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass  # skip corrupt/incomplete line
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
                    observation=Observation(
                        text=s.get("observation", ""),
                        available_actions=s.get("available_actions"),
                    ),
                    action=Action(text=s.get("action", ""),
                                  action_type=s.get("action_type", "text"),
                                  metadata=s.get("action_metadata", {})),
                    next_observation=(
                        Observation(
                            text=s["next_observation"],
                            available_actions=s.get("next_available_actions"),
                        )
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

        # Truncate to requested amounts to avoid over-collection
        if target_per_type:
            capped: list[Trajectory] = []
            type_counts: dict[str, int] = {}
            for t in trajectories:
                tt = t.task_type
                limit = target_per_type.get(tt, float("inf"))
                count = type_counts.get(tt, 0)
                if count < limit:
                    capped.append(t)
                    type_counts[tt] = count + 1
            trajectories = capped
        elif n > 0:
            trajectories = trajectories[:n]

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
