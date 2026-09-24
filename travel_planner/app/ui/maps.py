"""Folium maps for the city overview and the planned routes."""

from typing import List, Optional, Sequence, Tuple

import folium
import streamlit as st
from streamlit_folium import st_folium

from app.data.cities import CITIES, CITY_CATALOG

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


def build_route_map(route: List[str], line_color: str, label: str,
                    segment_costs: Optional[List[float]] = None,
                    transport: Optional[str] = None) -> Optional[folium.Map]:
    """Build a map of the route fitted to its stops, or None without known coordinates."""
    coordinates = [CITIES[city] for city in route if city in CITIES]
    if not coordinates:
        return None
    route_map = create_fitted_map(coordinates)
    add_route_markers(route_map, route, segment_costs, transport)
    folium.PolyLine(
        coordinates, color=line_color, weight=LINE_WEIGHT, opacity=0.9, tooltip=label,
        dash_array=TRANSPORT_LINE_DASHES.get(transport),
    ).add_to(route_map)
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
    st_folium(folium_map, key=key, height=MAP_HEIGHT, use_container_width=True, returned_objects=[])
