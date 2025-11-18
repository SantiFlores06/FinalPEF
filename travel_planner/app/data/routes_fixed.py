# app/data/routes_fixed.py
# Contiene los datos fijos de rutas para el grafo
# Formato: (origin, destination, cost, time, transport_type)

ROUTES_FIXED = [
    # --- Rutas de TREN ---
    ("Madrid", "Barcelona", 50, 3, "tren"),
    ("Madrid", "Valencia", 40, 4, "tren"),
    ("Madrid", "Sevilla", 60, 3.5, "tren"),
    ("Barcelona", "Valencia", 45, 3.5, "tren"),
    ("Valencia", "Sevilla", 55, 4, "tren"),
    ("París", "Roma", 150, 10, "tren"),
    ("París", "Berlín", 120, 8, "tren"),
    ("Roma", "Berlín", 140, 11, "tren"),

    # --- Rutas de AVIÓN ---
    ("Madrid", "París", 150, 2.5, "avión"),
    ("Madrid", "Roma", 160, 2.8, "avión"),
    ("Madrid", "Berlín", 180, 3, "avión"),
    ("Madrid", "Lisboa", 70, 1.5, "avión"),
    ("Barcelona", "París", 100, 2, "avión"),
    ("Barcelona", "Roma", 130, 1.5, "avión"),
    ("Barcelona", "Berlín", 140, 2.5, "avión"),
    ("Sevilla", "París", 130, 2.5, "avión"),
    ("Sevilla", "Lisboa", 60, 1, "avión"),
    ("París", "Roma", 80, 2, "avión"),
    ("París", "Berlín", 90, 1.5, "avión"),
    ("París", "Lisboa", 110, 2.5, "avión"),
    ("Roma", "Berlín", 100, 2.2, "avión"),

    # --- Rutas de AUTO/BUS ---
    ("Madrid", "Barcelona", 620, 8, "auto"),
    ("Madrid", "Valencia", 360, 4, "auto"),
    ("Madrid", "Sevilla", 530, 6, "auto"),
    ("Madrid", "Lisboa", 630, 7, "auto"),
    ("Barcelona", "Valencia", 350, 3.8, "auto"),
    ("Sevilla", "Lisboa", 400, 5, "auto"),
]