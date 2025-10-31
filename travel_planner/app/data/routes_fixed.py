"""
routes_fixed.py - Red completa de ciudades con tres modos de transporte.
Incluye rutas asimétricas (ida ≠ vuelta) para optimización realista.
"""

from typing import List, Tuple

# Estructura: (origen, destino, costo, tiempo, transporte)
ROUTES_FIXED: List[Tuple[str, str, float, float, str]] = [

    # ======================================================
    # 🚗 AUTO (rutas asimétricas, costos y tiempos variados)
    # ======================================================
    ("Madrid", "Barcelona", 48, 6.0, "auto"),
    ("Barcelona", "Madrid", 52, 6.2, "auto"),

    ("Madrid", "Valencia", 38, 4.4, "auto"),
    ("Valencia", "Madrid", 40, 4.6, "auto"),

    ("Madrid", "Sevilla", 60, 5.6, "auto"),
    ("Sevilla", "Madrid", 58, 5.4, "auto"),

    ("Madrid", "Lisboa", 55, 6.0, "auto"),
    ("Lisboa", "Madrid", 50, 5.5, "auto"),

    ("Madrid", "París", 110, 9.0, "auto"),
    ("París", "Madrid", 105, 8.8, "auto"),

    ("Madrid", "Roma", 145, 12.0, "auto"),
    ("Roma", "Madrid", 140, 11.8, "auto"),

    ("Madrid", "Berlín", 160, 13.0, "auto"),
    ("Berlín", "Madrid", 155, 12.7, "auto"),

    ("Barcelona", "Valencia", 42, 3.5, "auto"),
    ("Valencia", "Barcelona", 46, 3.8, "auto"),

    ("Barcelona", "Lisboa", 85, 8.0, "auto"),
    ("Lisboa", "Barcelona", 80, 7.8, "auto"),

    ("Barcelona", "París", 100, 8.0, "auto"),
    ("París", "Barcelona", 95, 7.7, "auto"),

    ("Barcelona", "Roma", 130, 11.0, "auto"),
    ("Roma", "Barcelona", 125, 10.5, "auto"),

    ("Barcelona", "Berlín", 145, 12.0, "auto"),
    ("Berlín", "Barcelona", 140, 11.8, "auto"),

    ("Valencia", "Lisboa", 75, 7.0, "auto"),
    ("Lisboa", "Valencia", 72, 6.8, "auto"),

    ("Valencia", "París", 115, 9.0, "auto"),
    ("París", "Valencia", 110, 8.7, "auto"),

    ("Valencia", "Roma", 135, 11.0, "auto"),
    ("Roma", "Valencia", 130, 10.7, "auto"),

    ("Valencia", "Berlín", 150, 12.0, "auto"),
    ("Berlín", "Valencia", 145, 11.8, "auto"),

    ("Sevilla", "Lisboa", 50, 4.0, "auto"),
    ("Lisboa", "Sevilla", 48, 3.8, "auto"),

    ("Sevilla", "París", 130, 10.0, "auto"),
    ("París", "Sevilla", 125, 9.8, "auto"),

    ("Sevilla", "Roma", 140, 11.0, "auto"),
    ("Roma", "Sevilla", 135, 10.8, "auto"),

    ("Sevilla", "Berlín", 165, 13.0, "auto"),
    ("Berlín", "Sevilla", 160, 12.8, "auto"),

    ("Lisboa", "París", 120, 9.0, "auto"),
    ("París", "Lisboa", 115, 8.7, "auto"),

    ("Lisboa", "Roma", 140, 11.0, "auto"),
    ("Roma", "Lisboa", 135, 10.6, "auto"),

    ("Lisboa", "Berlín", 160, 12.5, "auto"),
    ("Berlín", "Lisboa", 155, 12.3, "auto"),

    ("París", "Roma", 150, 10.5, "auto"),
    ("Roma", "París", 145, 10.2, "auto"),

    ("París", "Berlín", 120, 9.0, "auto"),
    ("Berlín", "París", 118, 8.8, "auto"),

    ("Roma", "Berlín", 140, 10.0, "auto"),
    ("Berlín", "Roma", 135, 9.8, "auto"),

    # ======================================================
    # ✈️ AVIÓN (más caro, más rápido)
    # ======================================================
    ("Madrid", "Barcelona", 120, 1.0, "avión"),
    ("Barcelona", "Madrid", 130, 1.1, "avión"),

    ("Madrid", "Valencia", 110, 0.9, "avión"),
    ("Valencia", "Madrid", 115, 1.0, "avión"),

    ("Madrid", "Sevilla", 130, 1.1, "avión"),
    ("Sevilla", "Madrid", 125, 1.0, "avión"),

    ("Madrid", "Lisboa", 140, 1.3, "avión"),
    ("Lisboa", "Madrid", 135, 1.2, "avión"),

    ("Madrid", "París", 160, 2.0, "avión"),
    ("París", "Madrid", 150, 1.9, "avión"),

    ("Madrid", "Roma", 190, 2.2, "avión"),
    ("Roma", "Madrid", 185, 2.1, "avión"),

    ("Madrid", "Berlín", 210, 2.6, "avión"),
    ("Berlín", "Madrid", 205, 2.5, "avión"),

    ("Barcelona", "París", 170, 2.0, "avión"),
    ("París", "Barcelona", 160, 1.9, "avión"),

    ("Barcelona", "Roma", 180, 2.5, "avión"),
    ("Roma", "Barcelona", 175, 2.4, "avión"),

    ("Barcelona", "Berlín", 200, 2.7, "avión"),
    ("Berlín", "Barcelona", 190, 2.6, "avión"),

    ("Barcelona", "Lisboa", 155, 1.8, "avión"),
    ("Lisboa", "Barcelona", 150, 1.7, "avión"),

    ("Lisboa", "París", 165, 2.1, "avión"),
    ("París", "Lisboa", 160, 2.0, "avión"),

    ("Lisboa", "Roma", 180, 2.3, "avión"),
    ("Roma", "Lisboa", 175, 2.2, "avión"),

    ("Lisboa", "Berlín", 210, 2.8, "avión"),
    ("Berlín", "Lisboa", 205, 2.7, "avión"),

    ("Valencia", "París", 175, 2.0, "avión"),
    ("París", "Valencia", 170, 1.9, "avión"),

    ("Valencia", "Roma", 190, 2.4, "avión"),
    ("Roma", "Valencia", 185, 2.3, "avión"),

    ("Valencia", "Berlín", 210, 2.6, "avión"),
    ("Berlín", "Valencia", 200, 2.5, "avión"),

    ("Sevilla", "París", 185, 2.3, "avión"),
    ("París", "Sevilla", 175, 2.2, "avión"),

    ("Sevilla", "Roma", 200, 2.5, "avión"),
    ("Roma", "Sevilla", 190, 2.4, "avión"),

    ("Sevilla", "Berlín", 220, 2.9, "avión"),
    ("Berlín", "Sevilla", 210, 2.8, "avión"),

    # ======================================================
    # 🚄 TREN (medio costo, medio tiempo)
    # ======================================================
    ("Madrid", "Barcelona", 70, 3.0, "tren"),
    ("Barcelona", "Madrid", 72, 3.1, "tren"),

    ("Madrid", "Valencia", 60, 2.5, "tren"),
    ("Valencia", "Madrid", 62, 2.6, "tren"),

    ("Madrid", "Sevilla", 65, 3.2, "tren"),
    ("Sevilla", "Madrid", 63, 3.0, "tren"),

    ("Madrid", "Lisboa", 75, 4.0, "tren"),
    ("Lisboa", "Madrid", 72, 3.8, "tren"),

    ("Madrid", "París", 95, 6.0, "tren"),
    ("París", "Madrid", 90, 5.8, "tren"),

    ("Madrid", "Roma", 120, 7.5, "tren"),
    ("Roma", "Madrid", 115, 7.2, "tren"),

    ("Madrid", "Berlín", 135, 8.5, "tren"),
    ("Berlín", "Madrid", 130, 8.3, "tren"),

    ("Barcelona", "Valencia", 55, 2.0, "tren"),
    ("Valencia", "Barcelona", 58, 2.1, "tren"),

    ("Barcelona", "París", 100, 5.0, "tren"),
    ("París", "Barcelona", 95, 4.8, "tren"),

    ("Barcelona", "Roma", 140, 8.0, "tren"),
    ("Roma", "Barcelona", 135, 7.8, "tren"),

    ("Barcelona", "Berlín", 150, 9.0, "tren"),
    ("Berlín", "Barcelona", 145, 8.8, "tren"),

    ("Lisboa", "París", 110, 6.0, "tren"),
    ("París", "Lisboa", 108, 5.9, "tren"),

    ("Lisboa", "Roma", 125, 7.0, "tren"),
    ("Roma", "Lisboa", 122, 6.9, "tren"),

    ("Lisboa", "Berlín", 135, 8.0, "tren"),
    ("Berlín", "Lisboa", 132, 7.8, "tren"),

    ("Valencia", "París", 110, 6.0, "tren"),
    ("París", "Valencia", 105, 5.8, "tren"),

    ("Valencia", "Roma", 130, 7.5, "tren"),
    ("Roma", "Valencia", 125, 7.3, "tren"),

    ("Valencia", "Berlín", 140, 8.5, "tren"),
    ("Berlín", "Valencia", 135, 8.3, "tren"),

    ("Sevilla", "Lisboa", 80, 4.5, "tren"),
    ("Lisboa", "Sevilla", 78, 4.4, "tren"),

    ("Sevilla", "París", 120, 6.5, "tren"),
    ("París", "Sevilla", 115, 6.3, "tren"),

    ("Sevilla", "Roma", 130, 7.2, "tren"),
    ("Roma", "Sevilla", 125, 7.0, "tren"),

    ("Sevilla", "Berlín", 140, 8.2, "tren"),
    ("Berlín", "Sevilla", 135, 8.0, "tren"),
]
