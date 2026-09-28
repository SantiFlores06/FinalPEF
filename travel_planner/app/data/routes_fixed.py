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
# Regular ferry and cruise lanes within one sea basin; only cruise hubs sail farther, across oceans.
MAX_SEA_LANE_KM = 5000
# Ships follow coasts and canals, so they sail farther than the great-circle distance.
SEA_DETOUR_FACTOR = 1.3
# Hubs fly direct to every other hub; a hub reaches regional airports up to long haul, while two
# regional airports only share short and medium-haul flights. Farther trips connect through hubs.
MAX_HUB_TO_REGIONAL_FLIGHT_KM = 6000
MAX_REGIONAL_FLIGHT_KM = 2000

Feasibility = Callable[[City, City, float], bool]


def overland_link(origin: City, destination: City, distance_km: float) -> bool:
    """Road and rail need both cities on the same landmass and within reach."""
    return origin.landmass == destination.landmass and distance_km <= MAX_OVERLAND_KM


def air_link(origin: City, destination: City, distance_km: float) -> bool:
    """Direct flights link any two hubs, and shorter distances as soon as a regional airport is involved."""
    if origin.air_hub and destination.air_hub:
        return True
    if origin.air_hub or destination.air_hub:
        return distance_km <= MAX_HUB_TO_REGIONAL_FLIGHT_KM
    return distance_km <= MAX_REGIONAL_FLIGHT_KM


def share_sea_basin(origin: City, destination: City) -> bool:
    return not set(origin.sea_basins).isdisjoint(destination.sea_basins)


def sea_link(origin: City, destination: City, distance_km: float) -> bool:
    """Ships link two ports on the same basin within reach, or two cruise hubs across any ocean."""
    if not (origin.is_port and destination.is_port):
        return False
    if origin.cruise_hub and destination.cruise_hub:
        return True
    return share_sea_basin(origin, destination) and distance_km <= MAX_SEA_LANE_KM


@dataclass(frozen=True)
class TransportProfile:
    name: str
    cost_per_km: float
    speed_kmh: float
    long_distance_km: float
    long_distance_factor: float
    is_feasible: Feasibility
    boarding_hours: float = 0.0
    distance_factor: float = 1.0
    fixed_fee: float = 0.0


# Overland and sea trips get pricier per km on long distances, so chaining legs can pay off.
# Flights pay a fixed fee per takeoff (airport taxes) and get cheaper per km on long haul,
# so a direct flight beats chaining several shorter ones.
TRANSPORT_PROFILES = (
    TransportProfile(
        "auto", cost_per_km=0.18, speed_kmh=80, long_distance_km=1500, long_distance_factor=1.6,
        is_feasible=overland_link,
    ),
    TransportProfile(
        "tren", cost_per_km=0.22, speed_kmh=120, long_distance_km=1200, long_distance_factor=1.5,
        is_feasible=overland_link,
    ),
    TransportProfile(
        "avión", cost_per_km=0.12, speed_kmh=700, long_distance_km=1500, long_distance_factor=0.8,
        is_feasible=air_link, boarding_hours=0.8, fixed_fee=70,
    ),
    TransportProfile(
        "barco", cost_per_km=0.12, speed_kmh=35, long_distance_km=3000, long_distance_factor=1.3,
        is_feasible=sea_link, boarding_hours=2.0, distance_factor=SEA_DETOUR_FACTOR,
    ),
)

TRANSPORT_TYPES = tuple(profile.name for profile in TRANSPORT_PROFILES)


def build_route(origin: str, destination: str, distance_km: float, profile: TransportProfile) -> tuple:
    """Build one (origin, destination, cost, time, transport) route for a transport profile."""
    travelled_km = distance_km * profile.distance_factor
    distance_cost = travelled_km * profile.cost_per_km
    if travelled_km > profile.long_distance_km:
        distance_cost *= profile.long_distance_factor
    cost = profile.fixed_fee + distance_cost
    hours = travelled_km / profile.speed_kmh + profile.boarding_hours
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
