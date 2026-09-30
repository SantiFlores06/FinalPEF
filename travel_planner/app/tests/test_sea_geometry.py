"""Tests for the searoute sea lanes of ship routes and their fallback on the maps."""

import sys
from unittest.mock import Mock, patch

import folium
import pytest

from app.data.cities import CITIES
from app.ui import maps, sea_geometry
from app.ui.sea_geometry import fetch_sea_route

RIO, BUENOS_AIRES = CITIES["Río de Janeiro"], CITIES["Buenos Aires"]
TOKYO, LOS_ANGELES = CITIES["Tokio"], CITIES["Los Ángeles"]
# searoute answers in [longitude, latitude] order, like any GeoJSON
RIO_BUENOS_AIRES_FEATURE = {
    "geometry": {"coordinates": [[-43.2, -22.9], [-43.1, -23.2], [-56.5, -35.1], [-58.4, -34.6]]},
    "properties": {"length": 2288.7, "units": "km"},
}
PORT_TOLERANCE_DEGREES = 1.0


@pytest.fixture(autouse=True)
def fresh_sea_cache():
    sea_geometry.compute_sea_route.clear()
    yield
    sea_geometry.compute_sea_route.clear()


def fake_searoute(feature=RIO_BUENOS_AIRES_FEATURE, **behaviour):
    return Mock(searoute=Mock(return_value=feature, **behaviour))


def route_lines(route_map):
    return [child for child in route_map._children.values() if isinstance(child, folium.PolyLine)]


def tooltip_texts(line):
    return [child.text for child in line._children.values() if isinstance(child, folium.Tooltip)]


def test_searoute_is_asked_in_longitude_latitude_order():
    library = fake_searoute()
    with patch.dict(sys.modules, {"searoute": library}):
        fetch_sea_route(["Río de Janeiro", "Buenos Aires"])

    library.searoute.assert_called_once_with([RIO[1], RIO[0]], [BUENOS_AIRES[1], BUENOS_AIRES[0]],
                                             append_orig_dest=True)


def test_geojson_points_become_latitude_longitude_pairs():
    with patch.dict(sys.modules, {"searoute": fake_searoute()}):
        sea_route = fetch_sea_route(["Río de Janeiro", "Buenos Aires"])

    assert sea_route.segments == [[(-22.9, -43.2), (-23.2, -43.1), (-35.1, -56.5), (-34.6, -58.4)]]
    assert sea_route.distance_km == pytest.approx(2288.7)


def test_each_leg_is_traced_and_lengths_add_up():
    library = fake_searoute()
    with patch.dict(sys.modules, {"searoute": library}):
        sea_route = fetch_sea_route(["Río de Janeiro", "Buenos Aires", "Río de Janeiro"])

    assert library.searoute.call_count == 2
    assert len(sea_route.segments) == 2
    assert sea_route.distance_km == pytest.approx(2 * 2288.7)


def test_same_stops_are_traced_only_once():
    library = fake_searoute()
    with patch.dict(sys.modules, {"searoute": library}):
        first = fetch_sea_route(["Río de Janeiro", "Buenos Aires"])
        second = fetch_sea_route(["Río de Janeiro", "Buenos Aires"])

    assert library.searoute.call_count == 1
    assert first == second


def test_a_single_stop_needs_no_trace():
    library = fake_searoute()
    with patch.dict(sys.modules, {"searoute": library}):
        assert fetch_sea_route(["Río de Janeiro"]) is None

    library.searoute.assert_not_called()


def test_missing_library_falls_back_to_none():
    with patch.dict(sys.modules, {"searoute": None}):
        assert fetch_sea_route(["Río de Janeiro", "Buenos Aires"]) is None


def test_failures_are_logged_and_not_cached(caplog):
    library = fake_searoute(side_effect=[ValueError("no path"), RIO_BUENOS_AIRES_FEATURE])
    with patch.dict(sys.modules, {"searoute": library}):
        assert fetch_sea_route(["Río de Janeiro", "Buenos Aires"]) is None
        assert fetch_sea_route(["Río de Janeiro", "Buenos Aires"]) is not None

    assert "No sea route" in caplog.text


def test_real_offline_route_from_rio_to_buenos_aires_follows_the_coast():
    pytest.importorskip("searoute")

    sea_route = fetch_sea_route(["Río de Janeiro", "Buenos Aires"])

    [path] = sea_route.segments
    assert len(path) > 2
    assert path[0] == pytest.approx(RIO) and path[-1] == pytest.approx(BUENOS_AIRES)
    assert path[1] == pytest.approx(RIO, abs=PORT_TOLERANCE_DEGREES)
    assert path[-2] == pytest.approx(BUENOS_AIRES, abs=PORT_TOLERANCE_DEGREES)
    assert all(-40 < latitude < -20 and -60 < longitude < -40 for latitude, longitude in path)
    assert 2000 < sea_route.distance_km < 2600


def test_real_transpacific_route_gets_a_copy_ending_at_the_real_destination():
    pytest.importorskip("searoute")

    route, copy = fetch_sea_route(["Tokio", "Los Ángeles"]).segments

    assert route[0] == pytest.approx(TOKYO)
    assert copy[-1] == pytest.approx(LOS_ANGELES)
    steps = [after[1] - before[1] for before, after in zip(route, route[1:])]
    assert all(abs(step) < 180 for step in steps)


def test_ship_map_follows_the_sea_lanes_dashed_with_its_distance():
    with patch.dict(sys.modules, {"searoute": fake_searoute()}):
        route_map = maps.build_route_map(["Río de Janeiro", "Buenos Aires"], "#000", "Ruta", transport="barco")

    [line] = route_lines(route_map)
    assert len(line.locations) == 4
    assert line.options["dashArray"] == maps.TRANSPORT_LINE_DASHES["barco"]
    assert tooltip_texts(line) == ["Ruta · ~2289 km por mar"]


def test_ship_map_falls_back_to_a_dashed_straight_line_when_searoute_fails():
    library = fake_searoute(side_effect=RuntimeError("broken graph"))
    with patch.dict(sys.modules, {"searoute": library}):
        route_map = maps.build_route_map(["Barcelona", "Atenas"], "#000", "Ruta", transport="barco")

    [line] = route_lines(route_map)
    assert [tuple(point) for point in line.locations] == [CITIES["Barcelona"], CITIES["Atenas"]]
    assert line.options["dashArray"] == maps.TRANSPORT_LINE_DASHES["barco"]
    assert tooltip_texts(line) == ["Ruta"]
