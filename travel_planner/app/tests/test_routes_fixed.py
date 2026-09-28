"""Tests for the world city catalog and the routes generated from it."""

from itertools import permutations

import pytest

from app.core.graph import TravelGraph
from app.data.cities import ASIA, CITIES, CITY_CATALOG, EUROPE, MAINLAND_EUROPEAN_CITIES
from app.data.routes_fixed import (
    MAX_HUB_TO_REGIONAL_FLIGHT_KM, MAX_OVERLAND_KM, MAX_REGIONAL_FLIGHT_KM, MAX_SEA_LANE_KM, ROUTES_FIXED,
    TOLL_EUR_PER_KM, TRANSPORT_PROFILES, build_route, haversine_km, leg_toll, toll_rate_per_km,
)

OVERLAND_TRANSPORTS = ("auto", "tren")
DIRECT_LINKS = {(origin, destination, transport) for origin, destination, _cost, _time, transport in ROUTES_FIXED}
AIR_HUBS = [name for name, city in CITY_CATALOG.items() if city.air_hub]


def links_of(transport):
    return [(origin, destination) for origin, destination, route_transport in DIRECT_LINKS if route_transport == transport]


def route_graph():
    graph = TravelGraph()
    for origin, destination, cost, hours, transport in ROUTES_FIXED:
        graph.add_route(origin, destination, cost, hours, transport)
    return graph


def test_catalog_covers_the_world():
    assert len(CITIES) >= 120
    assert {"Madrid", "Tokio", "Nueva York", "Sídney", "Buenos Aires", "El Cairo"} <= set(CITIES)


def test_hubs_fly_direct_to_every_other_hub():
    assert {"Madrid", "Tokio", "Nueva York", "Buenos Aires", "Dubái", "Sídney"} <= set(AIR_HUBS)
    assert all((origin, destination, "avión") in DIRECT_LINKS for origin, destination in permutations(AIR_HUBS, 2))


def test_regional_airports_fly_direct_only_within_reach():
    for origin, destination in links_of("avión"):
        origin_city, destination_city = CITY_CATALOG[origin], CITY_CATALOG[destination]
        if origin_city.air_hub and destination_city.air_hub:
            continue
        reach_km = MAX_HUB_TO_REGIONAL_FLIGHT_KM if origin_city.air_hub or destination_city.air_hub \
            else MAX_REGIONAL_FLIGHT_KM
        assert haversine_km(origin, destination) <= reach_km


def test_two_far_regional_airports_have_no_direct_flight():
    assert ("Sevilla", "Perth", "avión") not in DIRECT_LINKS
    assert ("Oporto", "Vilna", "avión") not in DIRECT_LINKS


def test_every_city_is_reachable_by_plane():
    reachable = route_graph().shortest_path_totals("Ushuaia", "cost", "avión")

    assert set(reachable) == set(CITIES)


def test_overland_routes_stay_on_one_landmass_and_within_reach():
    for transport in OVERLAND_TRANSPORTS:
        for origin, destination in links_of(transport):
            assert CITY_CATALOG[origin].landmass == CITY_CATALOG[destination].landmass
            assert haversine_km(origin, destination) <= MAX_OVERLAND_KM


def test_flights_charge_a_fixed_fee_and_cheaper_long_haul_km():
    plane = next(profile for profile in TRANSPORT_PROFILES if profile.name == "avión")

    short_haul = build_route("A", "B", 1000, plane)
    long_haul = build_route("A", "B", 10000, plane)

    assert short_haul[2] == round(plane.fixed_fee + 1000 * plane.cost_per_km)
    assert plane.long_distance_factor < 1
    assert long_haul[2] < plane.fixed_fee + 10000 * plane.cost_per_km


def test_island_city_without_neighbors_has_no_overland_routes():
    assert not any(origin == "Reikiavik" for origin, _destination in links_of("auto"))


def test_no_road_crosses_the_atlantic():
    assert ("Madrid", "Nueva York", "auto") not in DIRECT_LINKS


def test_benchmark_pool_is_directly_linked_overland():
    for transport in OVERLAND_TRANSPORTS:
        assert all(
            (origin, destination, transport) in DIRECT_LINKS
            for origin, destination in permutations(MAINLAND_EUROPEAN_CITIES, 2)
        )


def test_benchmark_pool_is_fully_connected_by_the_lab_transports():
    graph = route_graph()
    for transport in ("auto", "tren", "avión"):
        for origin in MAINLAND_EUROPEAN_CITIES:
            assert set(MAINLAND_EUROPEAN_CITIES) <= set(graph.shortest_path_totals(origin, "cost", transport))


def test_ship_links_mediterranean_ports():
    assert ("Barcelona", "Atenas", "barco") in DIRECT_LINKS


def test_inland_city_has_no_ship_routes():
    assert not any("Madrid" in link for link in links_of("barco"))


def test_cruise_hubs_cross_oceans():
    assert ("Lisboa", "Nueva York", "barco") in DIRECT_LINKS


def test_ship_routes_outside_hubs_share_a_basin_within_reach():
    for origin, destination in links_of("barco"):
        origin_city, destination_city = CITY_CATALOG[origin], CITY_CATALOG[destination]
        if origin_city.cruise_hub and destination_city.cruise_hub:
            continue
        assert set(origin_city.sea_basins) & set(destination_city.sea_basins)
        assert haversine_km(origin, destination) <= MAX_SEA_LANE_KM


def test_car_cost_includes_the_tolls_of_its_continent():
    car = next(profile for profile in TRANSPORT_PROFILES if profile.name == "auto")
    distance_km = haversine_km("Madrid", "Barcelona")

    route = build_route("Madrid", "Barcelona", distance_km, car)

    expected_tolls = round(distance_km * TOLL_EUR_PER_KM[EUROPE])
    assert leg_toll("Madrid", "Barcelona", "auto") == expected_tolls
    assert route[2] == round(distance_km * car.cost_per_km) + expected_tolls


def test_tolls_average_the_rates_of_both_continents():
    expected_rate = (TOLL_EUR_PER_KM[EUROPE] + TOLL_EUR_PER_KM[ASIA]) / 2

    assert toll_rate_per_km("Moscú", "Almaty") == pytest.approx(expected_rate)


def test_european_roads_charge_more_tolls_than_north_american_ones():
    assert toll_rate_per_km("Madrid", "Barcelona") > toll_rate_per_km("Nueva York", "Washington D. C.")


def test_only_cars_pay_tolls():
    assert leg_toll("Madrid", "Barcelona", "tren") == 0
    assert leg_toll("Madrid", "Barcelona", "avión") == 0
    assert leg_toll("Barcelona", "Atenas", "barco") == 0
