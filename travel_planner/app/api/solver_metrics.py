"""Bounded in-memory history of the route computations the server actually ran."""

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, List

DEFAULT_MAX_RUNS = 500


@dataclass(frozen=True)
class SolverRun:
    """One real (non-cached) execution of a route algorithm."""

    algorithm: str
    n_cities: int
    elapsed_ms: float
    total_cost: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def average(values: List[float]) -> float:
    return sum(values) / len(values)


def group_elapsed_ms(runs: Iterable[SolverRun], key: Callable[[SolverRun], Any]) -> Dict[Any, List[float]]:
    """Group the elapsed milliseconds of the runs by the given key function."""
    groups: Dict[Any, List[float]] = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run.elapsed_ms)
    return groups


def summarize_by_algorithm(runs: List[SolverRun]) -> Dict[str, Dict[str, float]]:
    """Return the run count, average and maximum elapsed time of every algorithm."""
    return {
        algorithm: {"runs": len(elapsed), "avg_elapsed_ms": average(elapsed), "max_elapsed_ms": max(elapsed)}
        for algorithm, elapsed in group_elapsed_ms(runs, lambda run: run.algorithm).items()
    }


def summarize_by_city_count(runs: List[SolverRun]) -> List[Dict[str, Any]]:
    """Return the run count and average elapsed time of every (algorithm, city count) pair."""
    groups = group_elapsed_ms(runs, lambda run: (run.algorithm, run.n_cities))
    return [
        {"algorithm": algorithm, "n_cities": n_cities, "runs": len(elapsed), "avg_elapsed_ms": average(elapsed)}
        for (algorithm, n_cities), elapsed in sorted(groups.items())
    ]


class SolverMetrics:
    """Thread-safe recorder of solver runs and cache hits, keeping only the latest runs."""

    def __init__(self, max_runs: int = DEFAULT_MAX_RUNS) -> None:
        self._runs: Deque[SolverRun] = deque(maxlen=max_runs)
        self._cache_hits = 0
        self._lock = threading.Lock()

    def record_run(self, algorithm: str, n_cities: int, elapsed_ms: float, total_cost: float) -> None:
        with self._lock:
            self._runs.append(SolverRun(algorithm, n_cities, elapsed_ms, total_cost))

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._cache_hits = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return the raw runs, newest first, together with their aggregates."""
        with self._lock:
            runs = list(self._runs)
            cache_hits = self._cache_hits
        return {
            "total_runs": len(runs),
            "cache_hits": cache_hits,
            "runs": [asdict(run) for run in reversed(runs)],
            "by_algorithm": summarize_by_algorithm(runs),
            "by_city_count": summarize_by_city_count(runs),
        }
