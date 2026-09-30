"""Sea lanes of ship routes from the offline searoute maritime network, with a silent fallback when it is
unavailable."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from app.data.cities import CITIES
from app.ui.route_geometry import unwrap_longitudes, world_copies

logger = logging.getLogger(__name__)

SEA_CACHE_ENTRIES = 256
MIN_WAYPOINTS = 2

Coordinate = Tuple[float, float]
SeaLeg = Tuple[List[Coordinate], float]


@dataclass(frozen=True)
class SeaRoute:
    """Sea-lane polylines of a ship route as (latitude, longitude) points, with its length over the sea."""

    segments: List[List[Coordinate]]
    distance_km: float


def to_longitude_latitude(coordinate: Coordinate) -> List[float]:
    """Return a (latitude, longitude) pair in the [longitude, latitude] order searoute expects."""
    latitude, longitude = coordinate
    return [longitude, latitude]


def trace_sea_leg(origin: str, destination: str) -> SeaLeg:
    """Return the sea lane between two ports, starting and ending at the cities, with its length in km.

    Longitudes are unwrapped from the origin, like the flight arcs, so a trans-Pacific leg stays continuous.
    """
    import searoute  # optional and slow to load: imported on the first ship map only

    feature = searoute.searoute(
        to_longitude_latitude(CITIES[origin]), to_longitude_latitude(CITIES[destination]), append_orig_dest=True
    )
    path = [(latitude, longitude) for longitude, latitude in feature["geometry"]["coordinates"]]
    return unwrap_longitudes(path), float(feature["properties"]["length"])


@st.cache_data(max_entries=SEA_CACHE_ENTRIES, show_spinner=False)
def compute_sea_route(stops: Tuple[str, ...]) -> SeaRoute:
    """Trace every leg of the stop sequence; only successful routes get cached."""
    legs = [trace_sea_leg(origin, destination) for origin, destination in zip(stops, stops[1:])]
    segments = [copy for path, _ in legs for copy in world_copies(path)]
    return SeaRoute(segments, sum(length for _, length in legs))


def fetch_sea_route(stops: Sequence[str]) -> Optional[SeaRoute]:
    """Return the sea route through the stops, or None (logging why) when searoute cannot provide it."""
    if len(stops) < MIN_WAYPOINTS:
        return None
    try:
        return compute_sea_route(tuple(stops))
    except Exception as error:  # a missing or failing optional library must only cost the sea lanes
        logger.warning("No sea route for %s (%s: %s)", " → ".join(stops), type(error).__name__, error)
    return None
