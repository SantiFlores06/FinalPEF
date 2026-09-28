"""HTTP client for the Travel Planner API."""

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UNPROCESSABLE_ENTITY = 422
# The first request of a transport runs Dijkstra from every city before the API caches it
MATRIX_TIMEOUT_SECONDS = 20


class ApiError(Exception):
    """Error reported by the API with a message meant for the user."""


def extract_error_detail(response: requests.Response) -> str:
    """Return the human-readable `detail` of an API error response."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text
    if isinstance(detail, list):
        return "; ".join(str(error.get("msg", error)) for error in detail)
    return str(detail)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_api_health() -> bool:
    """Return whether the API answers its health check."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_api_health() -> bool:
    """Return the cached API health, never caching a failure."""
    is_healthy = fetch_api_health()
    if not is_healthy:
        fetch_api_health.clear()
    return is_healthy


@st.cache_data(ttl=3600, show_spinner=False)
def get_matrix_from_api(transport_mode: str, optimize_by: str) -> Optional[Dict]:
    """Fetch the cost, time and legs of the best route (by the criterion) between every pair of cities."""
    try:
        response = requests.get(
            f"{API_URL}/routes/matrix",
            params={"transport": transport_mode, "optimize_by": optimize_by},
            timeout=MATRIX_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            return response.json()
        st.error("Error al obtener la matriz desde el backend")
        return None
    except requests.RequestException as error:
        st.error(f"Error de conexión: {error}")
        return None


def optimize_multi_destination(
    cities: List[str],
    cost_matrix: List[List[float]],
    return_to_start: bool,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """Solve the multi-destination route (TSP), raising ApiError on failure."""
    payload: Dict[str, Any] = {
        "cities": cities,
        "cost_matrix": cost_matrix,
        "return_to_start": return_to_start,
    }
    if algorithm is not None:
        payload["algorithm"] = algorithm
    try:
        response = requests.post(f"{API_URL}/routes/optimize-multi", json=payload, timeout=60)
    except requests.RequestException as error:
        raise ApiError(f"Error optimizando ruta: {error}") from error
    if response.status_code == UNPROCESSABLE_ENTITY:
        raise ApiError(extract_error_detail(response))
    if response.status_code != 200:
        raise ApiError(f"Error optimizando ruta: {extract_error_detail(response)}")
    return response.json()


def calculate_shortest_route(
    origin: str,
    destination: str,
    transport_type: str,
    optimize_by: str,
) -> Optional[Dict]:
    """Compute the shortest path between two cities (Dijkstra)."""
    try:
        response = requests.post(
            f"{API_URL}/routes/shortest",
            json={
                "origin": origin,
                "destination": destination,
                "optimize_by": optimize_by,
                "transport_type": transport_type,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Error calculando ruta mínima: {extract_error_detail(response)}")
        return None
    except requests.RequestException as error:
        st.error(f"Error calculando ruta mínima: {error}")
        return None


def compare_transports(origin: str, destination: str, optimize_by: str) -> Optional[Dict]:
    """Fetch the best route of every transport between two cities, or None when unavailable."""
    try:
        response = requests.get(
            f"{API_URL}/routes/transports",
            params={"origin": origin, "destination": destination, "optimize_by": optimize_by},
            timeout=10,
        )
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


def create_reservation(user_id: str, itinerary: Dict) -> Optional[Dict]:
    """Create a single reservation."""
    try:
        response = requests.post(
            f"{API_URL}/reservations",
            json={"user_id": user_id, "itinerary": itinerary},
            timeout=10,
        )
        if response.status_code != 200:
            st.error(f"Error creando reserva: {extract_error_detail(response)}")
            return None
        return response.json()
    except requests.RequestException as error:
        st.error(f"Error creando reserva: {error}")
        return None


def create_reservations_batch(payload: List[Dict]) -> Optional[Dict]:
    """Create a batch of reservations."""
    try:
        response = requests.post(f"{API_URL}/reservations/batch", json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        st.error(f"Error creando lote: {extract_error_detail(response)}")
        return None
    except requests.RequestException as error:
        st.error(f"Error enviando lote: {error}")
        return None


def get_user_reservations(user_id: str) -> List[Dict]:
    """Fetch the reservations of a user."""
    try:
        response = requests.get(f"{API_URL}/reservations/user/{user_id}", timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.RequestException:
        return []


def cancel_reservation(reservation_id: str) -> bool:
    """Cancel a reservation."""
    try:
        response = requests.delete(f"{API_URL}/reservations/{reservation_id}", timeout=10)
        return response.status_code == 200
    except requests.RequestException as error:
        st.error(f"Error al cancelar reserva: {error}")
        return False


def get_algorithm_stats() -> Dict:
    """Fetch the real solver runs recorded by the API and their aggregates."""
    try:
        response = requests.get(f"{API_URL}/stats/algorithms", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except requests.RequestException:
        return {}


@st.cache_data(ttl=10, show_spinner=False)
def get_system_stats() -> Dict:
    """Fetch the system statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except requests.RequestException:
        return {}
