"""Tests for the great-circle arcs of flights and how the maps draw them."""

from unittest.mock import patch

import folium
import pytest

from app.data.cities import CITIES
from app.ui import maps
from app.ui.route_geometry import ARC_MAX_SEGMENTS, great_circle_path, world_copies

MADRID, NEW_YORK = CITIES["Madrid"], CITIES["Nueva York"]
TOKYO, LOS_ANGELES = CITIES["Tokio"], CITIES["Los Ángeles"]


def longitudes(path):
    return [longitude for _, longitude in path]


def longitude_steps(path):
    return [after - before for before, after in zip(longitudes(path), longitudes(path)[1:])]


def test_arc_starts_and_ends_at_the_cities():
    path = great_circle_path(MADRID, NEW_YORK)

    assert path[0] == MADRID
    assert path[-1] == pytest.approx(NEW_YORK)


def test_transatlantic_arc_bulges_toward_the_pole():
    path = great_circle_path(MADRID, NEW_YORK)
    straight_midpoint_latitude = (MADRID[0] + NEW_YORK[0]) / 2

    arc_midpoint_latitude, _ = path[len(path) // 2]
    assert arc_midpoint_latitude > straight_midpoint_latitude + 5


def test_transpacific_arc_crosses_the_pacific_without_jumping():
    path = great_circle_path(TOKYO, LOS_ANGELES)

    assert all(0 < step < 180 for step in longitude_steps(path))
    assert max(longitudes(path)) > 180
    assert path[-1] == pytest.approx((LOS_ANGELES[0], LOS_ANGELES[1] + 360))


def test_longer_legs_get_more_points_up_to_a_cap():
    short_leg = great_circle_path(MADRID, CITIES["Barcelona"])
    long_leg = great_circle_path(TOKYO, LOS_ANGELES)

    assert 2 < len(short_leg) < len(long_leg) <= ARC_MAX_SEGMENTS + 1


def test_coincident_points_keep_the_straight_line():
    assert great_circle_path(MADRID, MADRID) == [MADRID, MADRID]


def test_arc_within_the_map_needs_no_copy():
    path = great_circle_path(MADRID, NEW_YORK)

    assert world_copies(path) == [path]


def test_transpacific_arc_gets_a_copy_ending_at_the_real_destination():
    arc, copy = world_copies(great_circle_path(TOKYO, LOS_ANGELES))

    assert arc[0] == TOKYO
    assert copy[-1] == pytest.approx(LOS_ANGELES)
    assert longitude_steps(copy) == pytest.approx(longitude_steps(arc))


def test_westbound_transpacific_arc_gets_a_copy_ending_at_the_real_destination():
    arc, copy = world_copies(great_circle_path(LOS_ANGELES, TOKYO))

    assert arc[0] == LOS_ANGELES
    assert copy[-1] == pytest.approx(TOKYO)


def route_lines(route_map):
    return [child for child in route_map._children.values() if isinstance(child, folium.PolyLine)]


def fitted_bounds(route_map):
    [fit] = [child for child in route_map._children.values() if isinstance(child, folium.FitBounds)]
    return fit.bounds


def test_flight_map_draws_one_arc_per_leg():
    route = ["Madrid", "Nueva York", "Madrid"]

    route_map = maps.build_route_map(route, "#000", "Tu ruta", transport="avión")

    outbound, inbound = route_lines(route_map)
    assert len(outbound.locations) > 2
    assert tuple(outbound.locations[0]) == MADRID and tuple(inbound.locations[-1]) == MADRID


def test_transpacific_flight_map_reaches_both_markers_and_fits_the_real_cities():
    route_map = maps.build_route_map(["Tokio", "Los Ángeles"], "#000", "Tu ruta", transport="avión")

    arc, copy = route_lines(route_map)
    assert tuple(arc.locations[0]) == TOKYO
    assert tuple(copy.locations[-1]) == pytest.approx(LOS_ANGELES)
    assert fitted_bounds(route_map) == [[LOS_ANGELES[0], LOS_ANGELES[1]], [TOKYO[0], TOKYO[1]]]


def test_flight_map_falls_back_to_a_straight_line_if_the_arc_fails():
    with patch.object(maps, "great_circle_path", side_effect=ValueError("broken arc")):
        route_map = maps.build_route_map(["Madrid", "Nueva York"], "#000", "Tu ruta", transport="avión")

    [line] = route_lines(route_map)
    assert [tuple(point) for point in line.locations] == [MADRID, NEW_YORK]
