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

        Each worker is a long-running loop that creates its own agent + env
        and runs episodes until the global target is met. The main thread
        monitors the count of valid (and optionally successful) trajectories
        in real time and signals workers to stop once n is reached.

        No pre-allocated oversample — workers run on demand until the
        exact target count is satisfied.

        Args:
            agent_factory: Callable that returns a fresh SEAAgent instance.
            env_factory: Callable that returns a fresh SEAEnv instance.
            n: Target number of valid trajectories.
            max_workers: Maximum concurrent worker threads.
            task_ids: Optional list of task IDs to cycle through.
            only_successful: If True, only count successful trajectories
                toward the target (failures are discarded).

        Returns:
            List of Trajectory objects (exactly n, or fewer if all workers exit).

        Example:
            trajectories = TrajectoryCollector.collect_parallel(
                agent_factory=lambda: SEAAgent(brain=LLMBrain(APIBackend(...)), ...),
                env_factory=lambda: TextCraftEnv(max_steps_val=15),
                n=100,
                max_workers=30,
                only_successful=True,
            )
        """
        import threading

        results: list[Trajectory] = []
        lock = threading.Lock()
        stop = threading.Event()
        stats = {"attempted": 0, "valid": 0}

        # Build task_id iterator per worker
        _task_ids = task_ids or [None]

        def _worker(worker_id: int) -> None:
            """Long-running worker: creates own agent+env, loops until stop."""
            agent = agent_factory()
            env = env_factory()
            episode = 0
            try:
                while not stop.is_set():
                    tid = _task_ids[episode % len(_task_ids)]
                    episode += 1
                    try:
                        traj = agent.run_episode(env, task_id=tid)
                        traj.metadata["env_name"] = env.name
                        traj.metadata["worker_id"] = worker_id
                    except Exception as e:
                        logger.debug("Worker %d episode failed: %s", worker_id, e)
                        continue

                    with lock:
                        stats["attempted"] += 1
                        if only_successful and not traj.success:
                            continue
                        results.append(traj)
                        stats["valid"] = len(results)
                        count = stats["valid"]

                    if count % 10 == 0:
                        logger.info(
                            "  Progress: %d/%d valid (%d attempted)",
                            count, n, stats["attempted"],
                        )
                    if count >= n:
                        stop.set()
                        return
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        actual_workers = min(max_workers, n)
        logger.info(
            "Starting parallel collection: target=%d, workers=%d, only_successful=%s",
            n, actual_workers, only_successful,
        )

        threads = []
        for i in range(actual_workers):
            t = threading.Thread(target=_worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        # Wait for all workers to finish (they stop when target is met)
        for t in threads:
            t.join()

        logger.info(
            "Parallel collection done: %d valid trajectories (%d attempted, %.1f%% yield)",
            len(results), stats["attempted"],
            100 * len(results) / max(stats["attempted"], 1),
        )
        return results[:n]

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
