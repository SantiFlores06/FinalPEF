# FinalPEF

Integrantes: Esteban Ghinamo, Nicolas Moresco y Santiago Flores

Profesora: Valeria Daniele

Tema: 
5. Sistema de Planificación de Viajes Multidestino

Descripción:
Un sistema que calcule itinerarios de viaje óptimos considerando tiempos, costos y restricciones de transporte.

Requisitos:

Optimización algorítmica (algoritmo de camino mínimo, programación dinámica estilo “viajante”).

Memoización para subrutas ya calculadas.

Caching de combinaciones de vuelos/hoteles más usados.

Concurrencia para procesar reservas de múltiples usuarios.

Batching para procesar reservas masivas.

Profiling y refactorización del código.

Testing de validación de itinerarios.

Interface Grafica

IA: recomendación de itinerarios según preferencias aprendidas.


Estructura: 
travel_planner/
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── graph.py                 # Dijkstra, representación de grafo
│   │   ├── tsp_dp.py                # Held-Karp (DP + bitmask + memoización)
│   │   ├── tsp_genetic.py           # TSP con algoritmo genético (heurístico)
│   │   ├── itinerary_validator.py   # validaciones y business rules
│   ├── data/
│   │   ├── routes_fixed.py          # rutas europeas generadas (Haversine)
│   ├── caches/
│   │   ├── lru_cache.py             # LRU + TTL en memoria
│   │   ├── redis_cache.py           # caché distribuido (Redis)
│   │   ├── cache_backend.py         # selector de backend con fallback a LRU
│   ├── booking/
│   │   ├── reservations.py          # lógica de reservas (async)
│   │   ├── batching.py              # procesamiento por lotes
│   ├── ai/
│   │   ├── gemini_recommendations.py # recomendaciones con Gemini
│   ├── api/
│   │   ├── server.py                # FastAPI endpoints
│   ├── ui/
│   │   ├── streamlit_app.py         # interfaz Streamlit
│   └── tests/
│       ├── test_graph.py
│       ├── test_tsp.py
│       ├── ...                      # (ver carpeta tests/ para la suite completa)
├── requirements.txt
└── pytest.ini
