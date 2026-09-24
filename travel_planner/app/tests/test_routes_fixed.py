"""Tests for the world city catalog and the routes generated from it."""

from itertools import permutations

from app.data.cities import CITIES, CITY_CATALOG, MAINLAND_EUROPEAN_CITIES
from app.data.routes_fixed import MAX_OVERLAND_KM, MAX_SEA_LANE_KM, ROUTES_FIXED, haversine_km

OVERLAND_TRANSPORTS = ("auto", "tren")
DIRECT_LINKS = {(origin, destination, transport) for origin, destination, _cost, _time, transport in ROUTES_FIXED}


def links_of(transport):
    return [(origin, destination) for origin, destination, route_transport in DIRECT_LINKS if route_transport == transport]


def test_catalog_covers_the_world():
    assert len(CITIES) >= 120
    assert {"Madrid", "Tokio", "Nueva York", "Sídney", "Buenos Aires", "El Cairo"} <= set(CITIES)


def test_planes_connect_every_pair_of_cities():
    assert len(links_of("avión")) == len(CITIES) * (len(CITIES) - 1)


def test_overland_routes_stay_on_one_landmass_and_within_reach():
    for transport in OVERLAND_TRANSPORTS:
        for origin, destination in links_of(transport):
            assert CITY_CATALOG[origin].landmass == CITY_CATALOG[destination].landmass
            assert haversine_km(origin, destination) <= MAX_OVERLAND_KM


def test_island_city_without_neighbors_has_no_overland_routes():
    assert not any(origin == "Reikiavik" for origin, _destination in links_of("auto"))


def test_no_road_crosses_the_atlantic():
    assert ("Madrid", "Nueva York", "auto") not in DIRECT_LINKS


def test_benchmark_pool_is_fully_connected_by_the_lab_transports():
    for transport in ("auto", "tren", "avión"):
        assert all(
            (origin, destination, transport) in DIRECT_LINKS
            for origin, destination in permutations(MAINLAND_EUROPEAN_CITIES, 2)
        )


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
