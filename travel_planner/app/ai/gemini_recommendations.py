"""City recommendations generated with the Google Gen AI SDK (Gemini)."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_OUTPUT_TOKENS = 400
MAX_PARALLEL_REQUESTS = 8

_client = None
_client_lock = threading.Lock()


class RecommendationCache:
    """Process-wide, thread-safe store of the recommendations that were generated successfully."""

    def __init__(self) -> None:
        self._recommendations: Dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, location: str) -> Optional[str]:
        with self._lock:
            return self._recommendations.get(location)

    def put(self, location: str, recommendations: str) -> None:
        with self._lock:
            self._recommendations[location] = recommendations

    def clear(self) -> None:
        with self._lock:
            self._recommendations.clear()


recommendation_cache = RecommendationCache()


def get_client():
    """Return the Gemini client, creating it on first use; None when it is unavailable."""
    global _client
    if not GEMINI_AVAILABLE:
        return None
    with _client_lock:
        if _client is None:
            _client = create_client()
    return _client


def create_client():
    """Create the Gemini client from GOOGLE_API_KEY; None when it is missing or invalid."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY is not set; recommendations are disabled.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as error:
        logger.error("Could not create the Gemini client: %s", error)
        return None


def build_prompt(location: str) -> str:
    """Return the short prompt asking for the must-see places of a location."""
    return (
        f"Eres un experto en viajes. Da 3 o 4 lugares imprescindibles en {location}.\n"
        "Formato: lista con viñetas, una línea por lugar: '- Nombre - descripción breve'.\n"
        "Responde solo la lista, sin introducción ni cierre."
    )


def request_recommendations(client, location: str) -> Optional[str]:
    """Ask Gemini for the recommendations of a location, returning None on any failure."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_prompt(location),
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as error:
        logger.error("Could not generate recommendations for %s: %s", location, error)
        return None
    text = response.text if response else None
    return text.strip() if text else None


def generate_city_recommendations(city: str, country: Optional[str] = None) -> Optional[str]:
    """Return the places worth visiting in a city, reusing any previous successful answer."""
    location = f"{city}, {country}" if country else city
    cached_recommendations = recommendation_cache.get(location)
    if cached_recommendations:
        return cached_recommendations
    client = get_client()
    if not client:
        return None
    recommendations = request_recommendations(client, location)
    if recommendations:
        recommendation_cache.put(location, recommendations)
    return recommendations


def generate_recommendations_for_cities(cities: Iterable[str]) -> Dict[str, Optional[str]]:
    """Return the recommendations of every city, asking Gemini for all of them in parallel."""
    unique_cities = list(dict.fromkeys(cities))
    if not unique_cities:
        return {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_REQUESTS, len(unique_cities))) as executor:
        return dict(zip(unique_cities, executor.map(generate_city_recommendations, unique_cities)))
