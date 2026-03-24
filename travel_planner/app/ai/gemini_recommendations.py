# travel_planner/app/ai/gemini_recommendations.py
"""
Generador de recomendaciones usando Google Gen AI SDK (nuevo).
Proporciona recomendaciones de lugares que visitar en ciudades.
"""

import os
from typing import List, Optional

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Cliente global (se inicializa una sola vez)
_client = None


def get_client():
    """Retorna el cliente de Gemini, inicializándolo si es necesario."""
    global _client

    if not GEMINI_AVAILABLE:
        return None

    if _client is not None:
        return _client

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY no configurada. Recomendaciones deshabilitadas.")
        return None

    try:
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar el cliente de Gemini: {e}")
        return None


# Modelo recomendado actualmente (estable y gratuito en el tier free)
GEMINI_MODEL = "gemini-2.5-flash-lite"
# para poner la api key usar el comando $env:GOOGLE_API_KEY="AIzaSyBc70S0O9SHAEH0G11XJPu9e4wX0X0-tko" en la terminal

def generate_city_recommendations(city: str, country: str = None) -> Optional[str]:
    """
    Genera recomendaciones de lugares a visitar en una ciudad usando Gemini.

    Args:
        city: Nombre de la ciudad
        country: País opcional (para desambiguar)

    Returns:
        Texto con recomendaciones o None si hay error.
    """
    client = get_client()
    if not client:
        return None

    try:
        location = f"{city}, {country}" if country else city

        prompt = f"""
Eres un asistente de viajes experto. Proporciona 3-4 recomendaciones breves y atractivas
de lugares que NO DEBEN PERDERSE en {location}.

Formato: Lista con viñetas, cada lugar en 1-2 líneas máximo.
Incluye: nombre del lugar y breve descripción.

Ejemplo:
- Sagrada Familia - Basílica icónica de Gaudí, imprescindible
- Casa Batlló - Otro masterpiece arquitectónico modernista

Responde SOLO las recomendaciones, sin introducción ni conclusión extra.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            return response.text.strip()
        return None

    except Exception as e:
        print(f"[ERROR] Error generando recomendaciones para {city}: {e}")
        return None


def generate_itinerary_summary(cities: List[str], total_cost: float, transport_mode: str) -> Optional[str]:
    """
    Genera un resumen motivacional del itinerario usando Gemini.

    Args:
        cities: Lista de ciudades en el itinerario
        total_cost: Costo total en euros
        transport_mode: Tipo de transporte (auto, avión, tren)

    Returns:
        Texto motivacional o None si hay error.
    """
    client = get_client()
    if not client:
        return None

    try:
        cities_str = " → ".join(cities)

        prompt = f"""
Eres un asistente de viajes entusiasta. Genera un comentario motivacional corto (2-3 líneas máximo)
sobre el itinerario del usuario:

Ruta: {cities_str}
Costo total: {total_cost}€
Transporte: {transport_mode}

Ejemplo: "¡Qué increíble itinerario! Visitarás 4 ciudades europeas por solo 500€. ¡A disfrutar!"

Responde SOLO el comentario, sin texto adicional.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            return response.text.strip()
        return None

    except Exception as e:
        print(f"[ERROR] Error generando resumen: {e}")
        return None