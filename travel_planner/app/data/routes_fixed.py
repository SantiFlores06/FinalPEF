# app/data/routes_fixed.py
# Generación automática de rutas FULL conectadas en Europa
# Formato: (origin, destination, cost, time, transport_type)

import math
from dataclasses import dataclass
from itertools import permutations

# ==========================================
# CIUDADES (coordenadas aproximadas reales)
# ==========================================
CITIES = {
    # Europa Occidental
    "Madrid": (40.4, -3.7),
    "Barcelona": (41.3, 2.1),
    "Valencia": (39.4, -0.3),
    "Sevilla": (37.3, -5.9),
    "Bilbao": (43.2, -2.9),
    "Lisboa": (38.7, -9.1),
    "Oporto": (41.1, -8.6),
    "París": (48.8, 2.3),
    "Lyon": (45.7, 4.8),
    "Burdeos": (44.8, -0.5),
    "Marsella": (43.3, 5.3),
    "Berlín": (52.5, 13.4),
    "Múnich": (48.1, 11.5),
    "Hamburgo": (53.5, 10.0),
    "Fráncfort": (50.1, 8.6),
    "Roma": (41.9, 12.5),
    "Milán": (45.4, 9.2),
    "Nápoles": (40.8, 14.2),
    "Florencia": (43.7, 11.2),
    "Venecia": (45.4, 12.3),
    "Londres": (51.5, -0.1),
    "Edimburgo": (55.9, -3.2),
    "Ámsterdam": (52.3, 4.9),
    "Bruselas": (50.8, 4.3),
    "Zúrich": (47.3, 8.5),
    "Ginebra": (46.2, 6.1),
    "Niza": (43.7, 7.2),
    "Dublín": (53.3, -6.2),
    # Europa del Norte
    "Copenhague": (55.6, 12.5),
    "Estocolmo": (59.3, 18.0),
    "Oslo": (59.9, 10.7),
    "Helsinki": (60.1, 24.9),
    "Tallin": (59.4, 24.7),
    "Riga": (56.9, 24.1),
    "Vilna": (54.7, 25.3),
    # Europa del Este
    "Viena": (48.2, 16.3),
    "Praga": (50.0, 14.4),
    "Varsovia": (52.2, 21.0),
    "Cracovia": (50.0, 19.9),
    "Budapest": (47.5, 19.0),
    "Bratislava": (48.1, 17.1),
    "Sofía": (42.7, 23.3),
    "Bucarest": (44.4, 26.1),
    "Zagreb": (45.8, 15.9),
    "Ljubljana": (46.0, 14.5),
    "Belgrado": (44.8, 20.4),
    "Kiev": (50.4, 30.5),
    # Europa del Sur / Mediterráneo
    "Atenas": (37.9, 23.7),
    "Estambul": (41.0, 28.9),
    "Sarajevo": (43.8, 18.3),
    "Tirana": (41.3, 19.8),
}

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
@dataclass(frozen=True)
class TransportProfile:
    name: str
    cost_per_km: float
    speed_kmh: float
    long_distance_km: float
    long_distance_surcharge: float
    boarding_hours: float = 0.0


# Long-distance surcharges make multi-leg itineraries worth comparing against direct ones.
TRANSPORT_PROFILES = (
    TransportProfile("auto", cost_per_km=0.18, speed_kmh=80, long_distance_km=1500, long_distance_surcharge=1.6),
    TransportProfile("tren", cost_per_km=0.22, speed_kmh=120, long_distance_km=1200, long_distance_surcharge=1.5),
    TransportProfile(
        "avión", cost_per_km=0.30, speed_kmh=700, long_distance_km=1500, long_distance_surcharge=1.7,
        boarding_hours=0.8,
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
    """Connect every ordered pair of cities with one route per transport profile."""
    routes = []
    for origin, destination in permutations(CITIES.keys(), 2):
        distance_km = haversine_km(origin, destination)
        routes.extend(build_route(origin, destination, distance_km, profile) for profile in TRANSPORT_PROFILES)
    return routes


ROUTES_FIXED = generate_routes()
