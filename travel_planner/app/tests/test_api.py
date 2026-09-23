"""Tests de integracion livianos para la API FastAPI."""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from app.api import server


@pytest.fixture()
def client():
    server.route_cache.clear()
    server.travel_graph.graph.clear()
    server.travel_graph.vertices.clear()
    server.reservation_manager.reservations.clear()
    return TestClient(server.app)


def test_health_endpoint_returns_system_stats(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "cache_stats" in payload
    assert "reservation_stats" in payload


def test_matrix_endpoint_validates_transport_and_metric(client):
    invalid = client.get("/routes/matrix", params={"transport": "barco"})
    assert invalid.status_code == 400

    response = client.get(
        "/routes/matrix",
        params={"transport": "tren", "optimize_by": "time"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "tren"
    assert payload["optimize_by"] == "time"
    assert len(payload["cities"]) == len(payload["matrix"])
    assert all(row[index] == 0.0 for index, row in enumerate(payload["matrix"]))


def test_shortest_route_endpoint_caches_second_request(client):
    request = {
        "origin": "Madrid",
        "destination": "Barcelona",
        "optimize_by": "cost",
        "transport_type": "auto",
    }

    first = client.post("/routes/shortest", json=request)
    second = client.post("/routes/shortest", json=request)

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["path"][0] == "Madrid"
    assert first_payload["path"][-1] == "Barcelona"
    assert first_payload["cached"] is False
    assert second_payload["cached"] is True
    assert second_payload["path"] == first_payload["path"]


def test_shortest_route_endpoint_filters_by_transport_and_allows_layover(client):
    request = {
        "origin": "Madrid",
        "destination": "Berlín",
        "optimize_by": "cost",
        "transport_type": "auto",
    }

    response = client.post("/routes/shortest", json=request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == ["Madrid", "Fráncfort", "Berlín"]
    assert payload["total_cost"] == 337


def test_shortest_route_cache_separates_transport_type(client):
    base_request = {
        "origin": "Madrid",
        "destination": "Berlín",
        "optimize_by": "cost",
    }

    auto = client.post(
        "/routes/shortest",
        json={**base_request, "transport_type": "auto"},
    )
    tren = client.post(
        "/routes/shortest",
        json={**base_request, "transport_type": "tren"},
    )

    assert auto.status_code == 200
    assert tren.status_code == 200
    assert auto.json()["path"] == ["Madrid", "Fráncfort", "Berlín"]
    assert tren.json()["path"] == ["Madrid", "Burdeos", "Fráncfort", "Berlín"]
    assert tren.json()["cached"] is False


FIVE_CITIES = ["Madrid", "Barcelona", "París", "Roma", "Berlín"]
FOURTEEN_CITIES = FIVE_CITIES + [
    "Lisboa", "Londres", "Viena", "Praga", "Ámsterdam",
    "Bruselas", "Zúrich", "Varsovia", "Atenas",
]


def plane_cost_matrix(client, cities):
    """Return the plane cost submatrix for the given cities from /routes/matrix."""
    payload = client.get("/routes/matrix", params={"transport": "avión", "optimize_by": "cost"}).json()
    indexes = [payload["cities"].index(city) for city in cities]
    matrix = payload["matrix"]
    return [[matrix[row][column] for column in indexes] for row in indexes]


def optimize_multi(client, cities, cost_matrix, **options):
    """Post a multi-destination optimization request."""
    return client.post(
        "/routes/optimize-multi",
        json={"cities": cities, "cost_matrix": cost_matrix, **options},
    )


def test_plane_cities_used_in_tsp_tests_are_fully_connected(client):
    matrix = plane_cost_matrix(client, FOURTEEN_CITIES)

    assert all(value != -1.0 for row in matrix for value in row)


def test_optimize_multi_uses_held_karp_for_few_cities(client):
    response = optimize_multi(client, FIVE_CITIES, plane_cost_matrix(client, FIVE_CITIES))

    assert response.status_code == 200
    payload = response.json()
    assert payload["algorithm"] == "held_karp"
    assert payload["optimal_route"][0] == payload["optimal_route"][-1] == "Madrid"
    assert payload["history"] is None
    assert payload["cached"] is False


def test_optimize_multi_uses_genetic_for_many_cities(client):
    response = optimize_multi(client, FOURTEEN_CITIES, plane_cost_matrix(client, FOURTEEN_CITIES))

    assert response.status_code == 200
    payload = response.json()
    assert payload["algorithm"] == "genetic"
    assert payload["history"]


def test_optimize_multi_honors_forced_genetic_algorithm(client):
    response = optimize_multi(
        client, FIVE_CITIES, plane_cost_matrix(client, FIVE_CITIES), algorithm="genetic"
    )

    assert response.status_code == 200
    assert response.json()["algorithm"] == "genetic"


def test_optimize_multi_rejects_too_many_cities(client):
    cities = ["Madrid"] * 26
    matrix = [[0.0] * 26 for _ in range(26)]

    response = optimize_multi(client, cities, matrix)

    assert response.status_code == 422


def test_optimize_multi_rejects_forced_held_karp_beyond_limit(client):
    response = optimize_multi(
        client, FOURTEEN_CITIES, plane_cost_matrix(client, FOURTEEN_CITIES), algorithm="held_karp"
    )

    assert response.status_code == 422


def test_optimize_multi_rejects_unknown_city(client):
    matrix = [[0.0, 100.0], [100.0, 0.0]]

    response = optimize_multi(client, ["Madrid", "Atlántida"], matrix)

    assert response.status_code == 422


def test_optimize_multi_rejects_matrix_size_mismatch(client):
    matrix = plane_cost_matrix(client, FIVE_CITIES[:3])

    response = optimize_multi(client, FIVE_CITIES, matrix)

    assert response.status_code == 422


def test_optimize_multi_rejects_disconnected_matrix_without_caching(client):
    cities = FIVE_CITIES[:3]
    matrix = [[0.0 if row == column else -1.0 for column in range(3)] for row in range(3)]

    first = optimize_multi(client, cities, matrix)
    second = optimize_multi(client, cities, matrix)

    assert first.status_code == 422
    assert second.status_code == 422


def test_optimize_multi_caches_second_identical_request(client):
    matrix = plane_cost_matrix(client, FIVE_CITIES)

    first = optimize_multi(client, FIVE_CITIES, matrix)
    second = optimize_multi(client, FIVE_CITIES, matrix)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["optimal_route"] == first.json()["optimal_route"]
