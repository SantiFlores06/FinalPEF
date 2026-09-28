"""Session state defaults and navigation state."""

import copy
import time

import streamlit as st

HOME_PAGE = "home"
ROUTE_PLANNER_PAGE = "route_planner"
RESERVATIONS_PAGE = "reservations"
STATISTICS_PAGE = "statistics"
LABORATORY_PAGE = "laboratory"

BATCH_SETTLE_SECONDS = 5.0

SESSION_DEFAULTS = {
    "user_id": "user_1",
    "page": HOME_PAGE,
    "selected_cities": [],
    "clear_selected_cities": False,
    "last_transport": None,
    "last_optimize_by": None,
    "city_recommendations": {},
    "user_route_result": None,
    "optimized_route_result": None,
    "selected_route_for_booking": None,
    "ga_result": None,
    "cost_submatrix": None,
    "transport_comparison": None,
    "batch_settle_deadline": 0.0,
    "reservations_auto_refresh": False,
}

ROUTE_RESULT_KEYS = (
    "user_route_result",
    "optimized_route_result",
    "selected_route_for_booking",
    "ga_result",
    "cost_submatrix",
    "transport_comparison",
)


def init_state() -> None:
    """Set every missing session key to its default and apply pending requests."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default)
    if st.session_state.clear_selected_cities:
        st.session_state.selected_cities = []
        st.session_state.clear_selected_cities = False
    if "pending_page" in st.session_state:
        st.session_state.page = st.session_state.pop("pending_page")


def reset_route_results() -> None:
    """Forget every computed route and the recommendations tied to them."""
    for key in ROUTE_RESULT_KEYS:
        st.session_state[key] = None
    st.session_state.city_recommendations = {}


def request_page(page_id: str) -> None:
    """Ask the next run to switch to another page."""
    st.session_state.pending_page = page_id


def mark_batch_submitted() -> None:
    """Remember that a batch was just queued, so its reservations are awaited for a while."""
    st.session_state.batch_settle_deadline = time.monotonic() + BATCH_SETTLE_SECONDS


def is_batch_settling() -> bool:
    """Return whether a recently queued batch may still be missing from the reservations."""
    return time.monotonic() < st.session_state.batch_settle_deadline
