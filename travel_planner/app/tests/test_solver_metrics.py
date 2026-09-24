"""Tests for the in-memory history of real solver runs."""

import pytest

from app.api.solver_metrics import KnownOptima, SolverMetrics, genetic_convergence, percentile, rolling_means


@pytest.fixture
def metrics():
    return SolverMetrics(max_runs=3)


def test_empty_history_has_no_runs_nor_aggregates(metrics):
    snapshot = metrics.snapshot()

    assert snapshot["total_runs"] == 0
    assert snapshot["runs"] == []
    assert snapshot["by_algorithm"] == {}
    assert snapshot["by_city_count"] == []
    assert snapshot["cache_hits"] == 0


def test_runs_are_listed_newest_first_with_their_fields(metrics):
    metrics.record_run("dijkstra", 2, 0.5, 120.0)
    metrics.record_run("held_karp", 5, 3.0, 900.0)

    runs = metrics.snapshot()["runs"]

    assert [run["algorithm"] for run in runs] == ["held_karp", "dijkstra"]
    assert runs[0]["n_cities"] == 5
    assert runs[0]["elapsed_ms"] == 3.0
    assert runs[0]["total_cost"] == 900.0
    assert runs[0]["timestamp"]


def test_aggregates_per_algorithm(metrics):
    metrics.record_run("held_karp", 5, 2.0, 900.0)
    metrics.record_run("held_karp", 7, 6.0, 1100.0)
    metrics.record_run("genetic", 14, 300.0, 2000.0)

    by_algorithm = metrics.snapshot()["by_algorithm"]

    assert by_algorithm["held_karp"] == {"runs": 2, "avg_elapsed_ms": 4.0, "max_elapsed_ms": 6.0}
    assert by_algorithm["genetic"]["runs"] == 1


def test_aggregates_per_algorithm_and_city_count(metrics):
    metrics.record_run("held_karp", 5, 2.0, 900.0)
    metrics.record_run("held_karp", 5, 4.0, 900.0)
    metrics.record_run("held_karp", 7, 6.0, 1100.0)

    by_city_count = metrics.snapshot()["by_city_count"]

    assert by_city_count == [
        {"algorithm": "held_karp", "n_cities": 5, "runs": 2, "avg_elapsed_ms": 3.0},
        {"algorithm": "held_karp", "n_cities": 7, "runs": 1, "avg_elapsed_ms": 6.0},
    ]


def test_history_keeps_only_the_latest_runs(metrics):
    for elapsed_ms in range(5):
        metrics.record_run("genetic", 14, float(elapsed_ms), 1.0)

    snapshot = metrics.snapshot()

    assert snapshot["total_runs"] == 3
    assert [run["elapsed_ms"] for run in snapshot["runs"]] == [4.0, 3.0, 2.0]


def test_cache_hits_are_counted_apart_from_runs(metrics):
    metrics.record_cache_hit()
    metrics.record_cache_hit()

    snapshot = metrics.snapshot()

    assert snapshot["cache_hits"] == 2
    assert snapshot["total_runs"] == 0


def test_clear_forgets_runs_and_cache_hits(metrics):
    metrics.record_run("dijkstra", 2, 1.0, 1.0)
    metrics.record_cache_hit()

    metrics.clear()

    assert metrics.snapshot()["total_runs"] == 0
    assert metrics.snapshot()["cache_hits"] == 0


def test_percentile_interpolates_between_values():
    values = [4.0, 1.0, 3.0, 2.0]

    assert percentile(values, 50) == 2.5
    assert percentile(values, 100) == 4.0
    assert percentile([7.0], 95) == 7.0


def test_rolling_means_use_at_most_the_window():
    assert rolling_means([2.0, 4.0, 6.0, 8.0], window=2) == [2.0, 3.0, 5.0, 7.0]


def test_elapsed_series_has_a_rolling_mean_per_algorithm():
    metrics = SolverMetrics()
    for elapsed_ms in (1.0, 3.0):
        metrics.record_run("dijkstra", 2, elapsed_ms, 10.0)
    metrics.record_run("genetic", 14, 100.0, 10.0)

    series = metrics.snapshot()["elapsed_series"]

    dijkstra_points = [point for point in series if point["algorithm"] == "dijkstra"]
    assert [point["rolling_mean_ms"] for point in dijkstra_points] == [1.0, 2.0]
    assert [point["run_number"] for point in dijkstra_points] == [1, 2]
    assert [point for point in series if point["algorithm"] == "genetic"][0]["rolling_mean_ms"] == 100.0


def test_latency_percentiles_per_algorithm():
    metrics = SolverMetrics()
    for elapsed_ms in (1.0, 2.0, 3.0, 4.0, 5.0):
        metrics.record_run("held_karp", 5, elapsed_ms, 10.0)

    assert metrics.snapshot()["latency_percentiles"]["held_karp"] == {"p50_ms": 3.0, "p95_ms": 4.8, "max_ms": 5.0}


def test_per_minute_counts_runs_and_cache_hits_with_their_hit_rate(metrics):
    metrics.record_run("dijkstra", 2, 1.0, 10.0)
    metrics.record_cache_hit()
    metrics.record_cache_hit()
    metrics.record_cache_hit()

    per_minute = metrics.snapshot()["per_minute"]

    assert sum(minute["runs"] for minute in per_minute) == 1
    assert sum(minute["cache_hits"] for minute in per_minute) == 3
    assert per_minute[-1]["hit_rate"] > 0


def test_efficiency_divides_time_and_cost_by_the_cities(metrics):
    metrics.record_run("held_karp", 4, 8.0, 400.0)

    assert metrics.snapshot()["efficiency"]["held_karp"] == {"avg_ms_per_city": 2.0, "avg_cost_per_city": 100.0}


def test_dijkstra_hops_are_summarized(metrics):
    metrics.record_run("dijkstra", 2, 1.0, 10.0, hops=1)
    metrics.record_run("dijkstra", 2, 1.0, 10.0, hops=3)

    assert metrics.snapshot()["dijkstra_hops"] == {"avg_hops": 2.0, "max_hops": 3}


def test_genetic_quality_lists_convergence_improvement_and_gap(metrics):
    metrics.record_run("held_karp", 5, 1.0, 90.0)
    metrics.record_run("genetic", 5, 50.0, 100.0, convergence_generation=7, improvement_percent=20.0, gap_percent=11.1)

    [quality] = metrics.snapshot()["genetic_quality"]

    assert quality["convergence_generation"] == 7
    assert quality["improvement_percent"] == 20.0
    assert quality["gap_percent"] == 11.1


def test_genetic_convergence_is_the_first_generation_reaching_the_final_best():
    history = [
        {"generation": 0, "best_cost": 200.0},
        {"generation": 1, "best_cost": 150.0},
        {"generation": 2, "best_cost": 150.0},
    ]

    assert genetic_convergence(history) == {"convergence_generation": 1, "improvement_percent": 25.0}


def test_genetic_convergence_without_history_is_unknown():
    assert genetic_convergence(None) == {"convergence_generation": None, "improvement_percent": None}


def test_known_optima_give_the_gap_only_for_solved_problems():
    optima = KnownOptima(max_problems=1)
    optima.remember("first", 100.0)
    optima.remember("second", 200.0)

    assert optima.gap_percent("second", 210.0) == 5.0
    assert optima.gap_percent("first", 100.0) is None


def test_clear_forgets_cache_hit_history(metrics):
    metrics.record_cache_hit()

    metrics.clear()

    assert metrics.snapshot()["per_minute"] == []
