# app/data/routes_fixed.py
# Automatic route generation between the cities of the world catalog
# Formato: (origin, destination, cost, time, transport_type)

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Callable

from app.data.cities import CITIES, CITY_CATALOG, City

# ==========================================
# DISTANCIA REAL (HAVERSINE)
# ==========================================
EARTH_RADIUS_KM = 6371


def haversine_km(origin: str, destination: str) -> float:
    """Great-circle distance in km between two catalog cities."""
    lat1, lon1 = CITIES[origin]
    lat2, lon2 = CITIES[destination]

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    half_chord = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(delta_lon / 2) ** 2
    )

    central_angle = 2 * math.atan2(math.sqrt(half_chord), math.sqrt(1 - half_chord))
    return EARTH_RADIUS_KM * central_angle


# ==========================================
# GENERADOR DE RUTAS (ficticias pero plausibles)
# ==========================================
# Road and rail links stop at this distance; longer trips must chain several legs.
MAX_OVERLAND_KM = 4000

Feasibility = Callable[[City, City, float], bool]


def overland_link(origin: City, destination: City, distance_km: float) -> bool:
    """Road and rail need both cities on the same landmass and within reach."""
    return origin.landmass == destination.landmass and distance_km <= MAX_OVERLAND_KM


def air_link(_origin: City, _destination: City, _distance_km: float) -> bool:
    """Planes connect any two cities."""
    return True


@dataclass(frozen=True)
class TransportProfile:
    name: str
    cost_per_km: float
    speed_kmh: float
    long_distance_km: float
    long_distance_surcharge: float
    is_feasible: Feasibility
    boarding_hours: float = 0.0


# Long-distance surcharges make multi-leg itineraries worth comparing against direct ones.
TRANSPORT_PROFILES = (
    TransportProfile(
        "auto", cost_per_km=0.18, speed_kmh=80, long_distance_km=1500, long_distance_surcharge=1.6,
        is_feasible=overland_link,
    ),
    TransportProfile(
        "tren", cost_per_km=0.22, speed_kmh=120, long_distance_km=1200, long_distance_surcharge=1.5,
        is_feasible=overland_link,
    ),
    TransportProfile(
        "avión", cost_per_km=0.30, speed_kmh=700, long_distance_km=1500, long_distance_surcharge=1.7,
        is_feasible=air_link, boarding_hours=0.8,
    ),
)


def build_route(origin: str, destination: str, distance_km: float, profile: TransportProfile) -> tuple:
    """Build one (origin, destination, cost, time, transport) route for a transport profile."""
    cost = distance_km * profile.cost_per_km
    if distance_km > profile.long_distance_km:
        cost *= profile.long_distance_surcharge
    hours = distance_km / profile.speed_kmh + profile.boarding_hours
    return origin, destination, round(cost), round(hours, 1), profile.name


def generate_routes() -> list:
    """Connect every ordered pair of cities with one route per transport able to link them."""
    routes = []
    for origin, destination in permutations(CITY_CATALOG, 2):
        distance_km = haversine_km(origin, destination)
        origin_city, destination_city = CITY_CATALOG[origin], CITY_CATALOG[destination]
        routes.extend(
            build_route(origin, destination, distance_km, profile)
            for profile in TRANSPORT_PROFILES
            if profile.is_feasible(origin_city, destination_city, distance_km)
        )
    return routes


ROUTES_FIXED = generate_routes()
