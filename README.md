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


## Arquitectura

Arquitectura en capas de tipo cliente-servidor:

```text
Streamlit (app/ui)  --HTTP-->  API FastAPI (app/api/server.py)  -->  algoritmos (app/core)
                                   |                                  Dijkstra, Held-Karp, genético
                                   +--> caches/  booking/  data/
```

- La interfaz Streamlit es un cliente liviano: pide rutas, reservas y estadísticas a la API por HTTP. Solo consulta directamente a Gemini (recomendaciones, en paralelo y con caché) y corre en su propio proceso el benchmark controlado del Laboratorio.
- El servidor elige el algoritmo del TSP en `POST /routes/optimize-multi`: Held-Karp hasta 12 ciudades (`HELD_KARP_MAX`), genético por encima y hasta 25 (`MAX_TSP_CITIES`; más ciudades devuelve 422). El campo opcional `algorithm` permite forzarlo. Los viajes de 2 ciudades usan `POST /routes/shortest` (Dijkstra).
- La API registra el tiempo real de cada corrida no cacheada (`app/api/solver_metrics.py`) y lo expone en `GET /stats/algorithms`; la página Laboratorio lo muestra como "Estadísticas de uso real" junto al "Benchmark controlado".

## Cómo ejecutarlo (Windows)

```powershell
cd travel_planner
.\install_and_run.bat
```

El script:

1. Crea el entorno virtual `.venv` si no existe.
2. Instala las dependencias solo si `requirements.txt` cambió desde la última instalación.
3. Verifica el entorno con `verify_setup.py`.
4. Inicia la API (puerto 8000) y la interfaz (puerto 8501) en la misma consola.
5. Comprueba los endpoints principales con `verify_setup.py --api` y abre el navegador.

- Interfaz: http://localhost:8501
- Documentación de la API: http://localhost:8000/docs
- `Ctrl+C` detiene todo.

Las recomendaciones con IA requieren la variable de entorno `GOOGLE_API_KEY` (por ejemplo en `travel_planner/.env`). Sin ella, el resto del sistema funciona normalmente.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Información básica de la API |
| GET | `/health` | Estado del servidor |
| GET | `/routes/matrix` | Matriz de costos/tiempos por transporte |
| POST | `/routes/shortest` | Camino mínimo entre dos ciudades (Dijkstra) |
| GET | `/routes/compare` | Ruta del usuario vs ruta óptima |
| POST | `/routes/optimize-multi` | TSP multidestino; el servidor elige Held-Karp o genético |
| POST | `/itinerary/plan` | Planificación con validaciónes |
| POST | `/reservations` | Crear una reserva |
| POST | `/reservations/batch` | Crear reservas por lotes |
| GET | `/reservations/{reservation_id}` | Consultar una reserva |
| GET | `/reservations/user/{user_id}` | Reservas de un usuario |
| DELETE | `/reservations/{reservation_id}` | Cancelar una reserva |
| GET | `/stats` | Estadísticas del sistema (caché, lotes) |
| GET | `/stats/algorithms` | Ejecuciones reales de los algoritmos |

## Tests

```powershell
cd travel_planner
.\.venv\Scripts\python.exe -m pytest
```

La suite (214 tests en `app/tests/`) también genera el reporte de cobertura en `htmlcov/index.html`.

## Estructura

```text
travel_planner/
├── install_and_run.bat          # instala (si hace falta) y levanta API + UI
├── verify_setup.py              # verifica entorno y endpoints (--api)
├── profiling_analysis.py        # profiling con cProfile
├── requirements.txt
├── pytest.ini
├── mypy.ini
├── .streamlit/config.toml
└── app/
    ├── core/
    │   ├── graph.py                 # grafo y Dijkstra
    │   ├── tsp_dp.py                # Held-Karp y selección de algoritmo (umbrales)
    │   ├── tsp_genetic.py           # TSP con algoritmo genético
    │   └── itinerary_validator.py   # reglas de validación de itinerarios
    ├── api/
    │   ├── server.py                # endpoints FastAPI
    │   └── solver_metrics.py        # registro de corridas reales
    ├── booking/
    │   ├── reservations.py          # reservas asincrónicas
    │   └── batching.py              # procesamiento por lotes
    ├── caches/
    │   ├── lru_cache.py             # LRU + TTL en memoria
    │   ├── redis_cache.py           # caché distribuido (Redis)
    │   └── cache_backend.py         # selector de backend con fallback a LRU
    ├── ai/
    │   └── gemini_recommendations.py # recomendaciones con Gemini (paralelo + caché)
    ├── data/
    │   ├── routes_fixed.py          # rutas europeas generadas (Haversine)
    │   └── users.json
    ├── ui/
    │   ├── streamlit_app.py         # punto de entrada de la interfaz
    │   ├── api_client.py            # cliente HTTP de la API
    │   ├── state.py, styles.py, formatting.py, maps.py, route_matrices.py
    │   └── views/                   # home, route_planner, reservations,
    │                                # statistics, laboratory, usage_stats
    └── tests/                       # suite de Pytest
```
