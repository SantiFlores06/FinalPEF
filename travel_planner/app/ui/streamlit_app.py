"""Entry point of the Travel Planner Streamlit UI."""

import os
import sys
from typing import Callable, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st  # noqa: E402

from app.ui.api_client import API_URL, check_api_health  # noqa: E402
from app.ui.state import (  # noqa: E402
    HOME_PAGE,
    LABORATORY_PAGE,
    RESERVATIONS_PAGE,
    ROUTE_PLANNER_PAGE,
    STATISTICS_PAGE,
    init_state,
)
from app.ui.styles import inject_styles  # noqa: E402
from app.ui.views.home import render_home  # noqa: E402
from app.ui.views.laboratory import render_laboratory  # noqa: E402
from app.ui.views.reservations import render_reservations  # noqa: E402
from app.ui.views.route_planner import render_route_planner  # noqa: E402
from app.ui.views.statistics import render_statistics  # noqa: E402


class Page(NamedTuple):
    """A navigation entry: its sidebar label and the view that renders it."""

    label: str
    render: Callable[[], None]


PAGES = {
    HOME_PAGE: Page("🏠 Inicio", render_home),
    ROUTE_PLANNER_PAGE: Page("🌍 Ruta Multidestino", render_route_planner),
    RESERVATIONS_PAGE: Page("📋 Mis Reservas", render_reservations),
    STATISTICS_PAGE: Page("📊 Estadísticas", render_statistics),
    LABORATORY_PAGE: Page("⚗️ Laboratorio", render_laboratory),
}


def page_label(page_id: str) -> str:
    """Return the navigation label of a page."""
    return PAGES[page_id].label


def render_sidebar() -> str:
    """Render the navigation sidebar and return the chosen page id."""
    with st.sidebar:
        st.header("✈️ Travel Planner")
        st.caption("Planificación de viajes multidestino")
        page_id = st.radio("Selecciona una opción:", list(PAGES), format_func=page_label, key="page")
        st.divider()
        st.caption(f"Usuario: {st.session_state.user_id[:12]}")
    return page_id


st.set_page_config(
    page_title="Travel Planner - Sistema de Planificación",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()
init_state()

if not check_api_health():
    st.error(f"La API no está disponible en {API_URL}. Verifica que esté ejecutándose.")
    st.stop()

PAGES[render_sidebar()].render()
