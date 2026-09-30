"""Tests de integracion livianos para la API FastAPI."""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from app.api import server
from app.data.cities import CITY_CATALOG
from app.data.routes_fixed import leg_toll


@pytest.fixture()
def client():
    server.route_cache.clear()
    server.travel_graph.graph.clear()
    server.travel_graph.vertices.clear()
    server.reservation_manager.reservations.clear()
    server.solver_metrics.clear()
    server.known_optima.clear()
    return TestClient(server.app)


def test_health_endpoint_returns_system_stats(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "cache_stats" in payload
    assert "reservation_stats" in payload


def test_matrix_endpoint_validates_transport_and_metric(client):
    invalid = client.get("/routes/matrix", params={"transport": "teletransporte"})
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
    assert payload["total_cost"] == 468


def test_car_route_reports_the_tolls_included_in_its_cost(client):
    request = {"origin": "Madrid", "destination": "Berlín", "optimize_by": "cost", "transport_type": "auto"}

    payload = client.post("/routes/shortest", json=request).json()

    expected_tolls = leg_toll("Madrid", "Fráncfort", "auto") + leg_toll("Fráncfort", "Berlín", "auto")
    assert payload["toll_cost"] == expected_tolls == 131
    assert payload["toll_cost"] < payload["total_cost"]


def test_routes_without_tolls_report_none(client):
    request = {"origin": "Madrid", "destination": "Berlín", "optimize_by": "cost", "transport_type": "tren"}

    assert client.post("/routes/shortest", json=request).json()["toll_cost"] is None


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


@pytest.mark.parametrize("origin, destination", [
    ("Madrid", "Tokio"),
    ("Buenos Aires", "Madrid"),
    ("Lisboa", "Moscú"),
])
def test_cheapest_long_haul_flight_is_direct(client, origin, destination):
    request = {"origin": origin, "destination": destination, "optimize_by": "cost", "transport_type": "avión"}

    response = client.post("/routes/shortest", json=request)

    assert response.status_code == 200
    assert response.json()["path"] == [origin, destination]


def shortest_flight(client, origin, destination, optimize_by):
    request = {"origin": origin, "destination": destination, "optimize_by": optimize_by, "transport_type": "avión"}
    return client.post("/routes/shortest", json=request)


def test_far_regional_airports_connect_through_hubs(client):
    payload = shortest_flight(client, "Sevilla", "Perth", "cost").json()

    layovers = payload["path"][1:-1]
    assert payload["path"][0] == "Sevilla" and payload["path"][-1] == "Perth"
    assert layovers
    assert all(CITY_CATALOG[city].air_hub for city in layovers)


def test_shortest_route_reports_cost_and_hours_whatever_the_criterion(client):
    for optimize_by in ("cost", "time"):
        payload = shortest_flight(client, "Oporto", "Vilna", optimize_by).json()

        legs = server.travel_graph.get_route_details(payload["path"], "avión")
        assert payload["optimize_by"] == optimize_by
        assert payload["total_cost"] == sum(leg["cost"] for leg in legs)
        assert payload["total_hours"] == round(sum(leg["time"] for leg in legs), 1)


def test_optimizing_by_time_can_choose_a_faster_but_pricier_path(client):
    by_cost = shortest_flight(client, "Oporto", "Vilna", "cost").json()
    by_time = shortest_flight(client, "Oporto", "Vilna", "time").json()

    assert by_time["path"] != by_cost["path"]
    assert by_time["cached"] is False
    assert by_time["total_hours"] < by_cost["total_hours"]
    assert by_time["total_cost"] > by_cost["total_cost"]


def test_shortest_route_rejects_unknown_criterion(client):
    assert shortest_flight(client, "Madrid", "Roma", "comfort").status_code == 400


@pytest.mark.parametrize("origin, destination", [
    ("Atlántida", "Madrid"),
    ("Madrid", "Atlántida"),
])
def test_shortest_route_rejects_unknown_city(client, origin, destination):
    response = shortest_flight(client, origin, destination, "cost")

    assert response.status_code == 422
    assert "Atlántida" in response.json()["detail"]


def test_shortest_route_returns_404_when_known_cities_are_not_connected(client):
    request = {"origin": "Madrid", "destination": "Nueva York", "optimize_by": "cost", "transport_type": "auto"}

    assert client.post("/routes/shortest", json=request).status_code == 404


def test_matrix_rejects_unknown_criterion(client):
    response = client.get("/routes/matrix", params={"transport": "avión", "optimize_by": "comfort"})

    assert response.status_code == 400


def matrix_value(payload, field, origin, destination):
    cities = payload["cities"]
    return payload[field][cities.index(origin)][cities.index(destination)]


def test_matrix_holds_the_best_itinerary_even_without_direct_flight(client):
    payload = client.get("/routes/matrix", params={"transport": "avión", "optimize_by": "cost"}).json()
    flight = shortest_flight(client, "Sevilla", "Perth", "cost").json()

    assert matrix_value(payload, "matrix", "Sevilla", "Perth") == flight["total_cost"]
    assert matrix_value(payload, "cost_matrix", "Sevilla", "Perth") == flight["total_cost"]
    assert matrix_value(payload, "time_matrix", "Sevilla", "Perth") == flight["total_hours"]
    assert matrix_value(payload, "legs_matrix", "Sevilla", "Perth") == len(flight["path"]) - 1


def test_time_matrix_follows_the_fastest_itineraries(client):
    payload = client.get("/routes/matrix", params={"transport": "avión", "optimize_by": "time"}).json()
    flight = shortest_flight(client, "Oporto", "Vilna", "time").json()

    assert matrix_value(payload, "matrix", "Oporto", "Vilna") == flight["total_hours"]
    assert matrix_value(payload, "cost_matrix", "Oporto", "Vilna") == flight["total_cost"]


def test_plane_matrix_connects_every_pair_of_cities(client):
    payload = client.get("/routes/matrix", params={"transport": "avión", "optimize_by": "cost"}).json()

    assert len(payload["cities"]) == len(CITY_CATALOG)
    assert all(value != -1.0 for row in payload["matrix"] for value in row)


def test_car_matrix_marks_other_landmasses_unreachable(client):
    payload = client.get("/routes/matrix", params={"transport": "auto", "optimize_by": "cost"}).json()

    assert matrix_value(payload, "matrix", "Madrid", "Nueva York") == -1.0
    assert matrix_value(payload, "legs_matrix", "Madrid", "Nueva York") == -1.0


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


def algorithm_stats(client):
    return client.get("/stats/algorithms").json()


def test_algorithm_stats_start_empty(client):
    response = client.get("/stats/algorithms")

    assert response.status_code == 200
    assert response.json()["total_runs"] == 0


def test_algorithm_stats_record_a_real_held_karp_run_once(client):
    matrix = plane_cost_matrix(client, FIVE_CITIES)

    optimize_multi(client, FIVE_CITIES, matrix)
    optimize_multi(client, FIVE_CITIES, matrix)

    stats = algorithm_stats(client)
    assert stats["total_runs"] == 1
    assert stats["cache_hits"] == 1
    assert stats["by_algorithm"]["held_karp"]["runs"] == 1
    assert stats["runs"][0]["n_cities"] == 5
    assert stats["by_city_count"] == [
        {"algorithm": "held_karp", "n_cities": 5, "runs": 1,
         "avg_elapsed_ms": stats["runs"][0]["elapsed_ms"]},
    ]


def test_algorithm_stats_ignore_infeasible_tsp_requests(client):
    cities = FIVE_CITIES[:3]
    matrix = [[0.0 if row == column else -1.0 for column in range(3)] for row in range(3)]

    optimize_multi(client, cities, matrix)

    assert algorithm_stats(client)["total_runs"] == 0


def test_algorithm_stats_record_dijkstra_for_shortest_routes(client):
    request = {"origin": "Madrid", "destination": "Barcelona", "optimize_by": "cost", "transport_type": "auto"}

    client.post("/routes/shortest", json=request)
    client.post("/routes/shortest", json=request)

    stats = algorithm_stats(client)
    assert stats["by_algorithm"]["dijkstra"]["runs"] == 1
    assert stats["runs"][0]["n_cities"] == 2
    assert stats["cache_hits"] == 1


def test_matrix_endpoint_serves_ship_routes_between_ports_only(client):
    payload = client.get("/routes/matrix", params={"transport": "barco", "optimize_by": "cost"}).json()

    assert "Barcelona" in payload["cities"]
    assert "Madrid" not in payload["cities"]


def test_shortest_route_endpoint_accepts_ship(client):
    request = {"origin": "Barcelona", "destination": "Atenas", "optimize_by": "time", "transport_type": "barco"}

    response = client.post("/routes/shortest", json=request)

    assert response.status_code == 200
    assert response.json()["path"][0] == "Barcelona"
    assert response.json()["path"][-1] == "Atenas"


def test_compare_endpoint_finds_a_direct_ship_route(client):
    response = client.get(
        "/routes/compare", params={"origin": "Barcelona", "destination": "Atenas", "transport": "barco"}
    )

    assert response.status_code == 200
    assert response.json()["direct_exists"] is True


def test_algorithm_stats_record_dijkstra_hops(client):
    client.post("/routes/shortest", json={
        "origin": "Madrid", "destination": "Berlín", "optimize_by": "cost", "transport_type": "auto",
    })

    stats = algorithm_stats(client)
    assert stats["runs"][0]["hops"] == 2
    assert stats["dijkstra_hops"] == {"avg_hops": 2.0, "max_hops": 2}


def test_genetic_run_after_held_karp_reports_its_gap_to_the_optimum(client):
    matrix = plane_cost_matrix(client, FIVE_CITIES)

    optimize_multi(client, FIVE_CITIES, matrix)
    optimize_multi(client, FIVE_CITIES, matrix, algorithm="genetic")

    [quality] = algorithm_stats(client)["genetic_quality"]
    assert quality["gap_percent"] >= 0
    assert quality["convergence_generation"] is not None
    assert quality["improvement_percent"] >= 0


def test_genetic_run_without_known_optimum_has_no_gap(client):
    optimize_multi(client, FIVE_CITIES, plane_cost_matrix(client, FIVE_CITIES), algorithm="genetic")

    assert algorithm_stats(client)["genetic_quality"][0]["gap_percent"] is None


def compare_transports(client, origin, destination):
    return client.get("/routes/transports", params={"origin": origin, "destination": destination})


def test_transport_comparison_marks_car_cheapest_and_plane_fastest(client):
    response = compare_transports(client, "Nueva York", "Washington D. C.")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cheapest"] == "auto"
    assert payload["fastest"] == "avión"
    options = {option["transport"]: option for option in payload["options"]}
    assert options["auto"]["total_cost"] < options["avión"]["total_cost"]
    assert options["avión"]["total_hours"] < options["auto"]["total_hours"]


def test_european_tolls_make_the_train_cheaper_than_the_car(client):
    payload = compare_transports(client, "Madrid", "Barcelona").json()

    options = {option["transport"]: option for option in payload["options"]}
    assert payload["cheapest"] == "tren"
    assert options["auto"]["total_cost"] - options["auto"]["toll_cost"] < options["tren"]["total_cost"]


def test_transport_comparison_reports_tolls_only_for_the_car(client):
    payload = compare_transports(client, "Madrid", "Barcelona").json()

    tolls = {option["transport"]: option["toll_cost"] for option in payload["options"]}
    assert tolls == {"auto": leg_toll("Madrid", "Barcelona", "auto"), "tren": None, "avión": None}
    assert tolls["auto"] > 0


def test_car_matrix_reports_the_tolls_of_each_itinerary(client):
    payload = client.get("/routes/matrix", params={"transport": "auto", "optimize_by": "cost"}).json()
    route = client.post("/routes/shortest", json={
        "origin": "Madrid", "destination": "Berlín", "optimize_by": "cost", "transport_type": "auto",
    }).json()

    assert matrix_value(payload, "toll_matrix", "Madrid", "Berlín") == route["toll_cost"]


def test_transport_comparison_sums_hours_along_multi_leg_paths(client):
    payload = compare_transports(client, "Madrid", "Berlín").json()

    car = next(option for option in payload["options"] if option["transport"] == "auto")
    assert car["path"] == ["Madrid", "Fráncfort", "Berlín"]
    legs = server.travel_graph.get_route_details(car["path"], "auto")
    assert car["total_hours"] == round(sum(leg["time"] for leg in legs), 1)
    assert car["total_cost"] == 468


def test_transport_comparison_omits_infeasible_transports(client):
    payload = compare_transports(client, "Madrid", "Nueva York").json()

    assert [option["transport"] for option in payload["options"]] == ["avión"]


def test_transport_comparison_rejects_unknown_city(client):
    assert compare_transports(client, "Atlántida", "Madrid").status_code == 422
