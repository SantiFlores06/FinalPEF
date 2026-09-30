"""Folium maps for the city overview and the planned routes."""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import folium
import streamlit as st
from streamlit_folium import st_folium

from app.data.cities import CITIES, CITY_CATALOG
from app.ui.road_geometry import RoadRoute, fetch_road_route
from app.ui.route_geometry import great_circle_path, world_copies
from app.ui.styles import map_frame

logger = logging.getLogger(__name__)

USER_ROUTE_COLOR = "#3B82F6"
OPTIMAL_ROUTE_COLOR = "#10B981"
GENETIC_ROUTE_COLOR = "#EC4899"
PORT_CITY_COLOR = "#0891B2"
LINE_WEIGHT = 4
MAP_HEIGHT = 420
MAP_TILES = "OpenStreetMap"
BOUNDS_PADDING = (30, 30)
POPUP_STYLE = "min-width:130px"
POPUP_NOTE_STYLE = "color:#6B7280;font-size:11px"
ENDPOINT_ICON = "home"
DEFAULT_STOP_ICON = "circle"
TRANSPORT_STOP_ICONS = {"auto": "car", "tren": "train", "avión": "plane", "barco": "ship"}
TRANSPORT_LINE_DASHES = {"barco": "10 8"}

Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class RouteLine:
    """Polyline pieces drawn for a route, sharing the tooltip that describes them."""

    segments: List[List[Coordinate]]
    tooltip: str


LineStrategy = Callable[[List[str], str], RouteLine]


def compute_bounds(coordinates: Sequence[Coordinate]) -> List[List[float]]:
    """Return the south-west and north-east corners enclosing the coordinates."""
    latitudes = [latitude for latitude, _ in coordinates]
    longitudes = [longitude for _, longitude in coordinates]
    return [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]


def create_fitted_map(coordinates: Sequence[Coordinate]) -> folium.Map:
    """Create a light map zoomed to fit the coordinates."""
    route_map = folium.Map(tiles=MAP_TILES)
    route_map.fit_bounds(compute_bounds(coordinates), padding=BOUNDS_PADDING)
    return route_map


def stop_popup_html(city: str, stop_number: int, next_segment_cost: Optional[float]) -> str:
    """Return the popup content of a route stop."""
    next_segment = ""
    if next_segment_cost is not None:
        next_segment = f"<br><span style='{POPUP_NOTE_STYLE}'>Siguiente tramo: {next_segment_cost:.0f} €</span>"
    return (
        f"<div style='{POPUP_STYLE}'><b>{city}</b>"
        f"<br><span style='{POPUP_NOTE_STYLE}'>Parada #{stop_number}</span>{next_segment}</div>"
    )


def stop_icon(is_endpoint: bool, transport: Optional[str]) -> folium.Icon:
    """Return the home icon for the endpoints and the transport icon for the other stops."""
    if is_endpoint:
        return folium.Icon(color="red", icon=ENDPOINT_ICON, prefix="fa")
    return folium.Icon(color="blue", icon=TRANSPORT_STOP_ICONS.get(transport, DEFAULT_STOP_ICON), prefix="fa")


def add_stop_marker(route_map: folium.Map, city: str, stop_number: int, is_endpoint: bool,
                    next_segment_cost: Optional[float], transport: Optional[str]) -> None:
    """Place the marker of one route stop on the map."""
    folium.Marker(
        CITIES[city],
        popup=folium.Popup(stop_popup_html(city, stop_number, next_segment_cost), max_width=200),
        tooltip=f"#{stop_number} {city}",
        icon=stop_icon(is_endpoint, transport),
    ).add_to(route_map)


def add_route_markers(route_map: folium.Map, route: List[str],
                      segment_costs: Optional[List[float]], transport: Optional[str]) -> None:
    """Mark each stop once, in route order, flagging the start and a closing return."""
    visited = set()
    stop_number = 0
    for position, city in enumerate(route):
        is_start = position == 0
        is_return = position == len(route) - 1 and city == route[0]
        if city not in CITIES or (city in visited and not is_return):
            continue
        stop_number += 1
        has_next_segment = segment_costs is not None and position < len(segment_costs) and not is_return
        next_segment_cost = segment_costs[position] if has_next_segment else None
        add_stop_marker(route_map, city, stop_number, is_start or is_return, next_segment_cost, transport)
        visited.add(city)


def road_tooltip(label: str, road_route: RoadRoute) -> str:
    """Return the line tooltip with the real driving distance and duration."""
    return f"{label} · {road_route.distance_km:.0f} km por carretera · {road_route.duration_hours:.1f} h"


def straight_line(stops: List[str], label: str) -> RouteLine:
    """Join the stops with straight lines: the fallback of every other geometry."""
    return RouteLine([[CITIES[city] for city in stops]], label)


def road_line(stops: List[str], label: str) -> RouteLine:
    """Draw car routes along the roads when OSRM provides them, otherwise a straight line."""
    road_route = fetch_road_route(stops)
    if road_route is None:
        return straight_line(stops, label)
    return RouteLine([road_route.coordinates], road_tooltip(label, road_route))


def flight_line(stops: List[str], label: str) -> RouteLine:
    """Draw each flight leg as a great-circle arc, copied across the antimeridian when it crosses it."""
    arcs = [
        arc
        for origin, destination in zip(stops, stops[1:])
        for arc in world_copies(great_circle_path(CITIES[origin], CITIES[destination]))
    ]
    return RouteLine(arcs, label) if arcs else straight_line(stops, label)


# Geometry of each transport's line; the others ("tren", "barco") keep the straight line
LINE_STRATEGIES: Dict[Optional[str], LineStrategy] = {
    "auto": road_line,
    "avión": flight_line,
}


def plan_route_line(stops: List[str], label: str, transport: Optional[str]) -> RouteLine:
    """Return the transport's line, falling back to straight lines if its geometry fails for any reason."""
    draw_line = LINE_STRATEGIES.get(transport, straight_line)
    try:
        return draw_line(stops, label)
    except Exception:  # a broken geometry must never break the whole map
        logger.exception("Could not draw the %s geometry, using straight lines", transport)
        return straight_line(stops, label)


def add_route_line(route_map: folium.Map, stops: List[str], line_color: str, label: str,
                   transport: Optional[str]) -> None:
    """Draw the route with the geometry of its transport."""
    route_line = plan_route_line(stops, label, transport)
    for segment in route_line.segments:
        folium.PolyLine(
            segment, color=line_color, weight=LINE_WEIGHT, opacity=0.9, tooltip=route_line.tooltip,
            dash_array=TRANSPORT_LINE_DASHES.get(transport),
        ).add_to(route_map)


def build_route_map(route: List[str], line_color: str, label: str,
                    segment_costs: Optional[List[float]] = None,
                    transport: Optional[str] = None) -> Optional[folium.Map]:
    """Build a map of the route fitted to its stops, or None without known coordinates."""
    stops = [city for city in route if city in CITIES]
    if not stops:
        return None
    route_map = create_fitted_map([CITIES[city] for city in stops])
    add_route_markers(route_map, route, segment_costs, transport)
    add_route_line(route_map, stops, line_color, label, transport)
    return route_map


def build_home_map() -> folium.Map:
    """Build the overview map with every available city, painting the ports apart."""
    home_map = create_fitted_map(list(CITIES.values()))
    for city, catalog_city in CITY_CATALOG.items():
        marker_color = PORT_CITY_COLOR if catalog_city.is_port else USER_ROUTE_COLOR
        folium.CircleMarker(
            location=catalog_city.coordinate,
            radius=6,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.75,
            tooltip=f"{city} (puerto)" if catalog_city.is_port else city,
            popup=folium.Popup(f"<b>{city}</b>", max_width=120),
        ).add_to(home_map)
    return home_map


def show_map(folium_map: Optional[folium.Map], key: str) -> None:
    """Render a map at the standard height, or a notice when there is none."""
    if folium_map is None:
        st.caption("No se encontraron coordenadas para las ciudades seleccionadas.")
        return
    with map_frame():
        st_folium(folium_map, key=key, height=MAP_HEIGHT, use_container_width=True, returned_objects=[])
