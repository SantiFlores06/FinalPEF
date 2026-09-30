"""Tests for the OSRM road geometry of car routes and its fallback on the maps, with HTTP mocked."""

from unittest.mock import Mock, patch

import folium
import pytest
import requests

from app.data.cities import CITIES
from app.ui import maps, road_geometry
from app.ui.road_geometry import (
    OSRM_ROUTE_PARAMS,
    OSRM_TIMEOUT_SECONDS,
    OSRM_URL,
    RoadRoute,
    build_route_url,
    fetch_road_route,
)

MADRID_BARCELONA_GEOJSON = [[-3.7, 40.4], [-1.0, 41.6], [2.1, 41.3]]
OK_ANSWER = {
    "code": "Ok",
    "routes": [{"geometry": {"coordinates": MADRID_BARCELONA_GEOJSON}, "distance": 621400.0, "duration": 21960.0}],
}
ROAD_ROUTE = RoadRoute([(40.4, -3.7), (41.6, -1.0), (41.3, 2.1)], distance_km=621.4, duration_hours=6.1)


@pytest.fixture(autouse=True)
def fresh_osrm_state():
    road_geometry.request_road_route.clear()
    road_geometry.osrm_cooldown.reset()
    yield
    road_geometry.request_road_route.clear()
    road_geometry.osrm_cooldown.reset()


def osrm_answer(payload):
    return Mock(status_code=200, json=Mock(return_value=payload))


def test_url_lists_every_stop_as_longitude_then_latitude():
    assert build_route_url(["Madrid", "Barcelona", "París"]) == (
        f"{OSRM_URL}/route/v1/driving/-3.7,40.4;2.1,41.3;2.3,48.8"
    )


def test_fetch_asks_once_for_the_full_geojson_route_with_a_short_timeout():
    with patch.object(road_geometry.requests, "get", return_value=osrm_answer(OK_ANSWER)) as http_get:
        fetch_road_route(["Madrid", "Barcelona", "Madrid"])

    http_get.assert_called_once_with(
        build_route_url(["Madrid", "Barcelona", "Madrid"]), params=OSRM_ROUTE_PARAMS, timeout=OSRM_TIMEOUT_SECONDS
    )
    assert OSRM_ROUTE_PARAMS == {"overview": "full", "geometries": "geojson"}


def test_geojson_points_become_latitude_longitude_pairs():
    with patch.object(road_geometry.requests, "get", return_value=osrm_answer(OK_ANSWER)):
        road_route = fetch_road_route(["Madrid", "Barcelona"])

    assert road_route.coordinates == [(40.4, -3.7), (41.6, -1.0), (41.3, 2.1)]
    assert road_route.distance_km == pytest.approx(621.4)
    assert road_route.duration_hours == pytest.approx(6.1)


def test_same_stops_are_requested_only_once():
    with patch.object(road_geometry.requests, "get", return_value=osrm_answer(OK_ANSWER)) as http_get:
        first = fetch_road_route(["Madrid", "Barcelona"])
        second = fetch_road_route(["Madrid", "Barcelona"])

    assert http_get.call_count == 1
    assert first == second


@pytest.mark.parametrize("payload", [
    {"code": "NoRoute", "message": "Impossible route between points"},
    {"code": "Ok", "routes": []},
    {"code": "Ok", "routes": [{"distance": 1.0}]},
])
def test_answers_without_a_usable_route_fall_back_to_none(payload):
    with patch.object(road_geometry.requests, "get", return_value=osrm_answer(payload)):
        assert fetch_road_route(["Madrid", "Barcelona"]) is None


def test_invalid_json_falls_back_to_none():
    answer = Mock(status_code=502, json=Mock(side_effect=ValueError("not json")))
    with patch.object(road_geometry.requests, "get", return_value=answer):
        assert fetch_road_route(["Madrid", "Barcelona"]) is None


def test_failures_are_not_cached():
    answers = [osrm_answer({"code": "NoRoute"}), osrm_answer(OK_ANSWER)]
    with patch.object(road_geometry.requests, "get", side_effect=answers):
        assert fetch_road_route(["Madrid", "Barcelona"]) is None
        assert fetch_road_route(["Madrid", "Barcelona"]) is not None


@pytest.mark.parametrize("network_error", [requests.Timeout("slow"), requests.ConnectionError("offline")])
def test_network_failure_pauses_osrm_instead_of_blocking_every_map(network_error):
    with patch.object(road_geometry.requests, "get", side_effect=network_error) as http_get:
        assert fetch_road_route(["Madrid", "Barcelona"]) is None
        assert fetch_road_route(["Roma", "Milán"]) is None

    assert http_get.call_count == 1


def test_a_single_stop_needs_no_request():
    with patch.object(road_geometry.requests, "get") as http_get:
        assert fetch_road_route(["Madrid"]) is None

    http_get.assert_not_called()


def route_lines(route_map):
    return [child for child in route_map._children.values() if isinstance(child, folium.PolyLine)]


def tooltip_texts(line):
    return [child.text for child in line._children.values() if isinstance(child, folium.Tooltip)]


def test_car_map_follows_the_road_and_shows_its_distance():
    with patch.object(maps, "fetch_road_route", return_value=ROAD_ROUTE):
        route_map = maps.build_route_map(["Madrid", "Barcelona"], "#000", "Tu ruta", transport="auto")

    [line] = route_lines(route_map)
    assert [tuple(point) for point in line.locations] == ROAD_ROUTE.coordinates
    assert tooltip_texts(line) == ["Tu ruta · 621 km por carretera · 6.1 h"]


def test_car_map_falls_back_to_a_straight_line_without_road_geometry():
    with patch.object(maps, "fetch_road_route", return_value=None):
        route_map = maps.build_route_map(["Madrid", "Barcelona"], "#000", "Tu ruta", transport="auto")

    [line] = route_lines(route_map)
    assert [tuple(point) for point in line.locations] == [CITIES["Madrid"], CITIES["Barcelona"]]


@pytest.mark.parametrize("transport", ["avión", "barco", "tren"])
def test_other_transports_never_ask_for_road_geometry(transport):
    with patch.object(maps, "fetch_road_route") as fetch:
        maps.build_route_map(["Barcelona", "Atenas"], "#000", "Ruta", transport=transport)

    fetch.assert_not_called()


def test_ship_map_keeps_a_dashed_straight_line():
    route_map = maps.build_route_map(["Barcelona", "Atenas"], "#000", "Ruta", transport="barco")

    [line] = route_lines(route_map)
    assert [tuple(point) for point in line.locations] == [CITIES["Barcelona"], CITIES["Atenas"]]
    assert line.options["dashArray"] == maps.TRANSPORT_LINE_DASHES["barco"]
