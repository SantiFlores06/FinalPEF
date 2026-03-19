# app/data/routes_fixed.py
# Generación automática de rutas FULL conectadas en Europa
# Formato: (origin, destination, cost, time, transport_type)

import math
from itertools import permutations

# ==========================================
# CIUDADES (coordenadas aproximadas)
# ==========================================
CITIES = {
    "Madrid": (40.4, -3.7),
    "Barcelona": (41.3, 2.1),
    "Valencia": (39.4, -0.3),
    "Sevilla": (37.3, -5.9),
    "Lisboa": (38.7, -9.1),
    "París": (48.8, 2.3),
    "Berlín": (52.5, 13.4),
    "Roma": (41.9, 12.5),
    "Londres": (51.5, -0.1),
    "Ámsterdam": (52.3, 4.9),
    "Bruselas": (50.8, 4.3),
    "Viena": (48.2, 16.3),
    "Praga": (50.0, 14.4),
    "Varsovia": (52.2, 21.0),
    "Budapest": (47.5, 19.0),
    "Copenhague": (55.6, 12.5),
    "Estocolmo": (59.3, 18.0),
    "Oslo": (59.9, 10.7),
    "Helsinki": (60.1, 24.9),
    "Zúrich": (47.3, 8.5),
    "Atenas": (37.9, 23.7),
    "Sofía": (42.7, 23.3),
    "Bucarest": (44.4, 26.1),
    "Zagreb": (45.8, 15.9),
    "Belgrado": (44.8, 20.4),
    "Kiev": (50.4, 30.5),
    "Estambul": (41.0, 28.9),
    "Dublín": (53.3, -6.2),
}

# ==========================================
# DISTANCIA REAL (HAVERSINE)
# ==========================================
def distancia(c1, c2):
    lat1, lon1 = CITIES[c1]
    lat2, lon2 = CITIES[c2]

    R = 6371  # radio de la tierra en km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # km


# ==========================================
# GENERADOR DE RUTAS
# ==========================================
def generar_rutas():
    rutas = []

    for origen, destino in permutations(CITIES.keys(), 2):
        d = distancia(origen, destino)

        # =========================
        # AUTO 🚗
        # =========================
        costo_auto = d * 0.18
        tiempo_auto = d / 80

        # penalizar distancias largas
        if d > 1500:
            costo_auto *= 1.6

        rutas.append((
            origen,
            destino,
            round(costo_auto),
            round(tiempo_auto, 1),
            "auto"
        ))

        # =========================
        # TREN 🚆
        # =========================
        costo_tren = d * 0.22
        tiempo_tren = d / 120

        if d > 1200:
            costo_tren *= 1.5  # hace que convenga hacer escalas

        rutas.append((
            origen,
            destino,
            round(costo_tren),
            round(tiempo_tren, 1),
            "tren"
        ))

        # =========================
        # AVIÓN ✈️
        # =========================
        costo_avion = d * 0.30
        tiempo_avion = d / 700 + 0.8  # incluye check-in

        # 🔥 clave: vuelos largos más caros (para Dijkstra)
        if d > 1500:
            costo_avion *= 1.7

        rutas.append((
            origen,
            destino,
            round(costo_avion),
            round(tiempo_avion, 1),
            "avión"
        ))

    return rutas


# ==========================================
# RESULTADO FINAL
# ==========================================
ROUTES_FIXED = generar_rutas()