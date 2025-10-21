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
│   │   ├── graph.py               # Dijkstra, representación de grafo
│   │   ├── tsp_dp.py              # DP estilo viajante con memoización
│   │   ├── itinerary_validator.py # validaciones y business rules
│   ├── caches/
│   │   ├── lru_cache.py
│   │   ├── redis_cache.py
│   ├── booking/
│   │   ├── reservations.py        # lógica reservas (async)
│   │   ├── batching.py
│   ├── api/
│   │   ├── server.py              # FastAPI/Flask endpoints
│   ├── ui/
│   │   ├── streamlit_app.py       # GUI rápida
│   ├── ml/
│   │   ├── recommender.py         # IA recomendador
│   └── tests/
│       ├── test_graph.py
│       ├── test_tsp.py
├── requirements.txt
├── Dockerfile
├── tox.ini / pytest.ini
└── README.md
