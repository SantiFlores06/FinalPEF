"""Home page."""

import streamlit as st

from app.data.routes_fixed import CITIES
from app.ui.maps import build_home_map, show_map
from app.ui.styles import card_container, render_page_header

FEATURES = [
    ("Ruta Multidestino", "Optimización TSP con Dijkstra, Held-Karp y Algoritmos Genéticos."),
    ("Mis Reservas", "Estado en tiempo real de reservas individuales y en lote."),
    ("Estadísticas", "Caché LRU, rendimiento y procesamiento batch asíncrono."),
    ("IA Gemini", "Recomendaciones personalizadas de destinos y lugares."),
]


def render_home() -> None:
    """Render the welcome page with the feature overview and the city map."""
    render_page_header(
        "Bienvenido al Planificador de Viajes",
        f"Optimizá rutas entre {len(CITIES)} ciudades del mundo, compará algoritmos de "
        "optimización y realizá reservas, todo en un solo lugar.",
    )
    for column, (title, description) in zip(st.columns(len(FEATURES)), FEATURES):
        with column, card_container():
            st.markdown(f"**{title}**")
            st.caption(description)
    st.divider()
    st.markdown(f"#### {len(CITIES)} ciudades disponibles en el mundo")
    st.caption("En celeste, las ciudades con puerto: se pueden conectar en barco o crucero.")
    show_map(build_home_map(), key="home_city_map")
