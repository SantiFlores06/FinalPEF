"""Bounded in-memory history of the route computations the server actually ran, and its aggregates."""

import math
import threading
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

DEFAULT_MAX_RUNS = 500
DEFAULT_MAX_CACHE_HITS = 2000
DEFAULT_MAX_KNOWN_OPTIMA = 200
ROLLING_WINDOW = 10
MEDIAN_PERCENT = 50
TAIL_PERCENT = 95
# ISO timestamps share the "YYYY-MM-DDTHH:MM" prefix within the same minute
MINUTE_PREFIX_LENGTH = len("YYYY-MM-DDTHH:MM")


def now_iso() -> str:
    return datetime.now().isoformat()


@dataclass(frozen=True)
class SolverRun:
    """One real (non-cached) execution of a route algorithm, with its optional quality measures."""

    algorithm: str
    n_cities: int
    elapsed_ms: float
    total_cost: float
    hops: Optional[int] = None
    convergence_generation: Optional[int] = None
    improvement_percent: Optional[float] = None
    gap_percent: Optional[float] = None
    timestamp: str = field(default_factory=now_iso)


def average(values: List[float]) -> float:
    return sum(values) / len(values)


def percentile(values: List[float], percent: float) -> float:
    """Return the linearly interpolated percentile of the values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rolling_means(values: List[float], window: int) -> List[float]:
    """Return the mean of each value together with up to window - 1 values before it."""
    means = []
    window_sum = 0.0
    for index, value in enumerate(values):
        window_sum += value
        if index >= window:
            window_sum -= values[index - window]
        means.append(window_sum / min(index + 1, window))
    return means


def minute_of(timestamp: str) -> str:
    return timestamp[:MINUTE_PREFIX_LENGTH]


def group_runs(runs: Iterable[SolverRun], key: Callable[[SolverRun], Any]) -> Dict[Any, List[SolverRun]]:
    """Group the runs by the given key function, keeping their order."""
    groups: Dict[Any, List[SolverRun]] = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return groups


def elapsed_of(runs: List[SolverRun]) -> List[float]:
    return [run.elapsed_ms for run in runs]


def by_algorithm(runs: List[SolverRun]) -> Dict[str, List[SolverRun]]:
    return group_runs(runs, lambda run: run.algorithm)


def summarize_by_algorithm(runs: List[SolverRun]) -> Dict[str, Dict[str, float]]:
    """Return the run count, average and maximum elapsed time of every algorithm."""
    return {
        algorithm: {"runs": len(group), "avg_elapsed_ms": average(elapsed_of(group)), "max_elapsed_ms": max(elapsed_of(group))}
        for algorithm, group in by_algorithm(runs).items()
    }


def summarize_by_city_count(runs: List[SolverRun]) -> List[Dict[str, Any]]:
    """Return the run count and average elapsed time of every (algorithm, city count) pair."""
    groups = group_runs(runs, lambda run: (run.algorithm, run.n_cities))
    return [
        {"algorithm": algorithm, "n_cities": n_cities, "runs": len(group), "avg_elapsed_ms": average(elapsed_of(group))}
        for (algorithm, n_cities), group in sorted(groups.items())
    ]


def summarize_latency_percentiles(runs: List[SolverRun]) -> Dict[str, Dict[str, float]]:
    """Return the median, tail and worst elapsed time of every algorithm."""
    return {
        algorithm: {
            "p50_ms": percentile(elapsed_of(group), MEDIAN_PERCENT),
            "p95_ms": percentile(elapsed_of(group), TAIL_PERCENT),
            "max_ms": max(elapsed_of(group)),
        }
        for algorithm, group in by_algorithm(runs).items()
    }


def build_elapsed_series(runs: List[SolverRun], window: int) -> List[Dict[str, Any]]:
    """Return every run in order with the rolling mean of its algorithm, to spot degradation."""
    series = []
    for algorithm, group in by_algorithm(runs).items():
        means = rolling_means(elapsed_of(group), window)
        series.extend(
            {"algorithm": algorithm, "run_number": number, "timestamp": run.timestamp,
             "elapsed_ms": run.elapsed_ms, "rolling_mean_ms": mean}
            for number, (run, mean) in enumerate(zip(group, means), start=1)
        )
    return sorted(series, key=lambda point: point["timestamp"])


def summarize_per_minute(runs: List[SolverRun], cache_hit_timestamps: List[str]) -> List[Dict[str, Any]]:
    """Return the runs, cache hits and cache hit rate of every minute with activity."""
    runs_per_minute = group_runs(runs, lambda run: minute_of(run.timestamp))
    hits_per_minute: Dict[str, int] = {}
    for timestamp in cache_hit_timestamps:
        hits_per_minute[minute_of(timestamp)] = hits_per_minute.get(minute_of(timestamp), 0) + 1
    summary = []
    for minute in sorted(set(runs_per_minute) | set(hits_per_minute)):
        run_count = len(runs_per_minute.get(minute, []))
        hit_count = hits_per_minute.get(minute, 0)
        summary.append({
            "minute": minute, "runs": run_count, "cache_hits": hit_count,
            "hit_rate": hit_count / (hit_count + run_count),
        })
    return summary


def summarize_efficiency(runs: List[SolverRun]) -> Dict[str, Dict[str, float]]:
    """Return the average elapsed time and cost per city of every algorithm."""
    return {
        algorithm: {
            "avg_ms_per_city": average([run.elapsed_ms / run.n_cities for run in group]),
            "avg_cost_per_city": average([run.total_cost / run.n_cities for run in group]),
        }
        for algorithm, group in by_algorithm(runs).items()
    }


def summarize_genetic_quality(runs: List[SolverRun]) -> List[Dict[str, Any]]:
    """Return the convergence, improvement and optimality gap of every genetic run."""
    return [
        {
            "timestamp": run.timestamp, "n_cities": run.n_cities,
            "convergence_generation": run.convergence_generation,
            "improvement_percent": run.improvement_percent, "gap_percent": run.gap_percent,
        }
        for run in runs if run.algorithm == "genetic"
    ]


def summarize_hops(runs: List[SolverRun]) -> Dict[str, float]:
    """Return the average and maximum number of legs of the Dijkstra routes."""
    hops = [run.hops for run in runs if run.hops is not None]
    if not hops:
        return {}
    return {"avg_hops": average(hops), "max_hops": max(hops)}


def genetic_convergence(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Optional[float]]:
    """Return the generation where the best cost stopped improving and the total improvement in %."""
    if not history:
        return {"convergence_generation": None, "improvement_percent": None}
    first_cost, final_cost = history[0]["best_cost"], history[-1]["best_cost"]
    convergence = next(entry["generation"] for entry in history if entry["best_cost"] == final_cost)
    improvement = None
    if math.isfinite(first_cost) and first_cost > 0:
        improvement = (first_cost - final_cost) / first_cost * 100
    return {"convergence_generation": convergence, "improvement_percent": improvement}


class KnownOptima:
    """Thread-safe, bounded memory of the exact optimum of recently solved problems."""

    def __init__(self, max_problems: int = DEFAULT_MAX_KNOWN_OPTIMA) -> None:
        self._optima: "OrderedDict[str, float]" = OrderedDict()
        self._max_problems = max_problems
        self._lock = threading.Lock()

    def remember(self, problem_key: str, optimal_cost: float) -> None:
        with self._lock:
            self._optima[problem_key] = optimal_cost
            self._optima.move_to_end(problem_key)
            if len(self._optima) > self._max_problems:
                self._optima.popitem(last=False)

    def gap_percent(self, problem_key: str, cost: float) -> Optional[float]:
        """Return how far above the known optimum a cost is, in %, or None when unknown."""
        with self._lock:
            optimal_cost = self._optima.get(problem_key)
        if not optimal_cost:
            return None
        return (cost - optimal_cost) / optimal_cost * 100

    def clear(self) -> None:
        with self._lock:
            self._optima.clear()


class SolverMetrics:
    """Thread-safe recorder of solver runs and cache hits, keeping only the latest ones."""

    def __init__(self, max_runs: int = DEFAULT_MAX_RUNS, max_cache_hits: int = DEFAULT_MAX_CACHE_HITS) -> None:
        self._runs: Deque[SolverRun] = deque(maxlen=max_runs)
        self._cache_hit_timestamps: Deque[str] = deque(maxlen=max_cache_hits)
        self._cache_hits = 0
        self._lock = threading.Lock()

    def record_run(self, algorithm: str, n_cities: int, elapsed_ms: float, total_cost: float, **quality: Any) -> None:
        """Record a run; quality holds the optional SolverRun measures (hops, convergence, gap...)."""
        run = SolverRun(algorithm, n_cities, elapsed_ms, total_cost, **quality)
        with self._lock:
            self._runs.append(run)

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1
            self._cache_hit_timestamps.append(now_iso())

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._cache_hit_timestamps.clear()
            self._cache_hits = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return the raw runs, newest first, together with their aggregates and time series."""
        with self._lock:
            runs = list(self._runs)
            cache_hit_timestamps = list(self._cache_hit_timestamps)
            cache_hits = self._cache_hits
        return {
            "total_runs": len(runs),
            "cache_hits": cache_hits,
            "runs": [asdict(run) for run in reversed(runs)],
            "by_algorithm": summarize_by_algorithm(runs),
            "by_city_count": summarize_by_city_count(runs),
            "rolling_window": ROLLING_WINDOW,
            "elapsed_series": build_elapsed_series(runs, ROLLING_WINDOW),
            "latency_percentiles": summarize_latency_percentiles(runs),
            "per_minute": summarize_per_minute(runs, cache_hit_timestamps),
            "efficiency": summarize_efficiency(runs),
            "genetic_quality": summarize_genetic_quality(runs),
            "dijkstra_hops": summarize_hops(runs),
        }
