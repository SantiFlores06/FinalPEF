"""Tests for the in-memory history of real solver runs."""

import pytest

from app.api.solver_metrics import SolverMetrics


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
