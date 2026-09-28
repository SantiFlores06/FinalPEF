"""Real road geometry of car routes from an OSRM server, with a silent fallback when it is unavailable."""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import streamlit as st

from app.data.cities import CITIES

logger = logging.getLogger(__name__)

OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org")
OSRM_TIMEOUT_SECONDS = 4
OSRM_ROUTE_PARAMS = {"overview": "full", "geometries": "geojson"}
OSRM_OK_CODE = "Ok"
# After a network failure OSRM is skipped for a while, so an offline server never delays every map
OSRM_COOLDOWN_SECONDS = 300
OSRM_CACHE_ENTRIES = 256
MIN_WAYPOINTS = 2
METERS_PER_KM = 1000
SECONDS_PER_HOUR = 3600

Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class RoadRoute:
    """Road polyline of a car route as (latitude, longitude) points, with its driving distance and duration."""

    coordinates: List[Coordinate]
    distance_km: float
    duration_hours: float


class RoadRouteUnavailable(Exception):
    """OSRM answered, but without a usable route."""


class Cooldown:
    """Remembers a recent failure for a while, thread-safely."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._until = 0.0
        self._lock = threading.Lock()

    def is_active(self) -> bool:
        with self._lock:
            return time.monotonic() < self._until

    def start(self) -> None:
        with self._lock:
            self._until = time.monotonic() + self._seconds

    def reset(self) -> None:
        with self._lock:
            self._until = 0.0


osrm_cooldown = Cooldown(OSRM_COOLDOWN_SECONDS)


def build_route_url(stops: Sequence[str]) -> str:
    """Return the OSRM driving route URL through every stop; OSRM expects each point as longitude,latitude."""
    waypoints = ";".join(f"{CITIES[city][1]},{CITIES[city][0]}" for city in stops)
    return f"{OSRM_URL}/route/v1/driving/{waypoints}"


def parse_road_route(payload: Dict[str, Any]) -> RoadRoute:
    """Return the first route of an OSRM answer, raising RoadRouteUnavailable when there is none."""
    routes = payload.get("routes") or []
    if payload.get("code") != OSRM_OK_CODE or not routes:
        raise RoadRouteUnavailable(payload.get("code"))
    route = routes[0]
    coordinates = [(latitude, longitude) for longitude, latitude in route["geometry"]["coordinates"]]
    return RoadRoute(coordinates, route["distance"] / METERS_PER_KM, route["duration"] / SECONDS_PER_HOUR)


@st.cache_data(max_entries=OSRM_CACHE_ENTRIES, show_spinner=False)
def request_road_route(stops: Tuple[str, ...]) -> RoadRoute:
    """Ask OSRM for the road route through every stop in one request; only successful answers get cached."""
    response = requests.get(build_route_url(stops), params=OSRM_ROUTE_PARAMS, timeout=OSRM_TIMEOUT_SECONDS)
    return parse_road_route(response.json())


def fetch_road_route(stops: Sequence[str]) -> Optional[RoadRoute]:
    """Return the road route through the stops, or None (logging why) when OSRM cannot provide it."""
    if len(stops) < MIN_WAYPOINTS or osrm_cooldown.is_active():
        return None
    try:
        return request_road_route(tuple(stops))
    except requests.RequestException as error:
        osrm_cooldown.start()
        logger.warning("OSRM is unreachable (%s), using straight lines for a while: %s", type(error).__name__, error)
    except (RoadRouteUnavailable, ValueError, KeyError, TypeError) as error:
        logger.warning("OSRM gave no road route for %s (%s: %s)", " → ".join(stops), type(error).__name__, error)
    return None
