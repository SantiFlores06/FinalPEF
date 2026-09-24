"""Short city tips generated with the Google Gen AI SDK (Gemini), one request per route."""

import json
import logging
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TIPS_PER_CITY = 3
OUTPUT_TOKENS_PER_CITY = 120
JSON_MIME_TYPE = "application/json"
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.5
TOO_MANY_REQUESTS = 429
FIRST_SERVER_ERROR = 500

# Interest shown to the user -> what the tips should focus on
INTEREST_FOCUS = {
    "Variado": "una mezcla de imperdibles, cultura, gastronomía y experiencias locales",
    "Cultura": "museos, historia, arquitectura y arte",
    "Gastronomía": "platos típicos, mercados y lugares para comer",
    "Naturaleza": "parques, miradores, playas y paseos al aire libre",
    "Vida nocturna": "bares, música en vivo y barrios para salir de noche",
    "Compras": "mercados, calles comerciales y artesanías locales",
}
INTERESTS = tuple(INTEREST_FOCUS)
DEFAULT_INTEREST = INTERESTS[0]

CityTips = List[str]

_client = None
_client_lock = threading.Lock()


class RecommendationCache:
    """Process-wide, thread-safe store of the tips generated successfully per (city, interest)."""

    def __init__(self) -> None:
        self._tips: Dict[Tuple[str, str], CityTips] = {}
        self._lock = threading.Lock()

    def get(self, city: str, interest: str) -> Optional[CityTips]:
        with self._lock:
            return self._tips.get((city, interest))

    def put(self, city: str, interest: str, tips: CityTips) -> None:
        with self._lock:
            self._tips[(city, interest)] = tips

    def clear(self) -> None:
        with self._lock:
            self._tips.clear()


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
        logger.error("Could not create the Gemini client (%s): %s", type(error).__name__, error)
        return None


def build_prompt(cities: List[str], interest: str) -> str:
    """Return the prompt asking for a few short tips per city, focused on the interest."""
    focus = INTEREST_FOCUS.get(interest, INTEREST_FOCUS[DEFAULT_INTEREST])
    return (
        f"Eres un experto en viajes. Para cada ciudad da exactamente {TIPS_PER_CITY} consejos "
        f"breves y variados (una línea, máximo 15 palabras cada uno) centrados en {focus}.\n"
        f"Ciudades: {'; '.join(cities)}.\n"
        "Responde en español. En 'city' repite el nombre de la ciudad tal como aparece en la lista."
    )


def build_response_schema():
    """Return the JSON schema of the answer: one object with its tips per city."""
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": types.Schema(type=types.Type.STRING),
                "tips": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            },
            required=["city", "tips"],
        ),
    )


def build_generation_config(city_count: int):
    """Return a config asking for structured JSON, with an output budget sized to the cities."""
    return types.GenerateContentConfig(
        max_output_tokens=OUTPUT_TOKENS_PER_CITY * city_count,
        response_mime_type=JSON_MIME_TYPE,
        response_schema=build_response_schema(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def is_retryable(error: Exception) -> bool:
    """Return whether the error is a rate limit or a server failure worth one more try."""
    status_code = getattr(error, "code", None)
    return isinstance(status_code, int) and (status_code == TOO_MANY_REQUESTS or status_code >= FIRST_SERVER_ERROR)


def generate_content_with_retry(client, cities: List[str], interest: str):
    """Call Gemini once, retrying after a short pause when it is rate limited or failing."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_prompt(cities, interest),
                config=build_generation_config(len(cities)),
            )
        except Exception as error:
            if attempt == MAX_RETRIES or not is_retryable(error):
                raise
            logger.warning("Gemini request failed (%s), retrying: %s", type(error).__name__, error)
            time.sleep(RETRY_BACKOFF_SECONDS)


def clean_tips(raw_tips) -> CityTips:
    """Keep the first non-empty tips, stripped."""
    if not isinstance(raw_tips, list):
        return []
    tips = [tip.strip() for tip in raw_tips if isinstance(tip, str) and tip.strip()]
    return tips[:TIPS_PER_CITY]


def parse_recommendations(text: Optional[str], cities: List[str]) -> Dict[str, CityTips]:
    """Return the tips of the requested cities found in the JSON answer; empty when malformed."""
    try:
        entries = json.loads(text or "")
    except ValueError:
        logger.warning("Gemini returned malformed JSON for %s", ", ".join(cities))
        return {}
    if not isinstance(entries, list):
        return {}
    requested = {city.casefold(): city for city in cities}
    recommendations = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("city"), str):
            continue
        city = requested.get(entry["city"].strip().casefold())
        tips = clean_tips(entry.get("tips"))
        if city and tips:
            recommendations[city] = tips
    return recommendations


def request_recommendations(client, cities: List[str], interest: str) -> Dict[str, CityTips]:
    """Ask Gemini for the tips of every city in a single call, returning {} on any failure."""
    try:
        response = generate_content_with_retry(client, cities, interest)
    except Exception as error:
        logger.error("Could not generate recommendations (%s): %s", type(error).__name__, error)
        return {}
    return parse_recommendations(response.text if response else None, cities)


def fetch_missing_recommendations(cities: List[str], interest: str) -> Dict[str, CityTips]:
    """Request the tips of cities not cached yet, caching the ones that arrive."""
    client = get_client()
    if not client:
        return {}
    recommendations = request_recommendations(client, cities, interest)
    for city, tips in recommendations.items():
        recommendation_cache.put(city, interest, tips)
    return recommendations


def generate_recommendations_for_cities(
    cities: Iterable[str], interest: str = DEFAULT_INTEREST
) -> Dict[str, Optional[CityTips]]:
    """Return the tips of every city (None when unavailable), requesting only the uncached ones."""
    unique_cities = list(dict.fromkeys(cities))
    recommendations = {city: recommendation_cache.get(city, interest) for city in unique_cities}
    missing_cities = [city for city, tips in recommendations.items() if tips is None]
    if missing_cities:
        recommendations.update(fetch_missing_recommendations(missing_cities, interest))
    return recommendations
