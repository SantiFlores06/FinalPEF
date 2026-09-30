"""
server.py - Servidor FastAPI que integra todos los módulos del sistema.
API RESTful para el sistema de planificación de viajes multidestino.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from datetime import datetime
from contextlib import asynccontextmanager, suppress
from app.data.routes_fixed import CITIES, ROUTES_FIXED, TOLLED_TRANSPORTS, TRANSPORT_TYPES, leg_toll
import asyncio
import copy
import hashlib
import logging
from functools import lru_cache
from time import perf_counter

from app.core.graph import PathTotals, TravelGraph
from app.core.tsp_dp import HELD_KARP_MAX, MAX_TSP_CITIES, TSPSolver, choose_tsp_algorithm
from app.core.tsp_genetic import GeneticTSP
from app.caches.cache_backend import get_cache_backend
from app.booking.reservations import ReservationManager
from app.booking.batching import ReservationBatchProcessor
from app.api.solver_metrics import KnownOptima, SolverMetrics, genetic_convergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN PRINCIPAL
# ==========================================================

BATCH_SIZE = 20
BATCH_TIMEOUT_SECONDS = 0.5
BATCH_TICK_SECONDS = 0.5
MAX_CONCURRENT_RESERVATIONS = BATCH_SIZE
SHORTEST_PATH_CITY_COUNT = 2
VALID_METRICS = {"cost", "time"}
VALID_TRANSPORTS = set(TRANSPORT_TYPES)
UNREACHABLE_MARKER = -1.0
HOURS_DECIMALS = 1


async def batch_loop():
    """Flush the reservation queue every tick so partial batches wait at most a tick."""
    while True:
        await asyncio.sleep(BATCH_TICK_SECONDS)
        await batch_processor.trigger_processing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando loop de procesamiento de lotes en background...")
    task = asyncio.create_task(batch_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Travel Planner API",
    description="Sistema de planificación de viajes multidestino con optimización algorítmica",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

travel_graph = TravelGraph()
route_cache = get_cache_backend(capacity=100)
solver_metrics = SolverMetrics()
known_optima = KnownOptima()
reservation_manager = ReservationManager(max_concurrent=MAX_CONCURRENT_RESERVATIONS)
batch_processor = ReservationBatchProcessor(
    batch_size=BATCH_SIZE,
    timeout_seconds=BATCH_TIMEOUT_SECONDS,
    reservation_manager=reservation_manager,
)

# ==========================================================
# MODELOS Pydantic
# ==========================================================


class RouteComparison(BaseModel):
    """Modelo para comparar rutas directa vs económica"""
    origin: str
    destination: str
    direct_route: Optional[Dict] = None  # Ruta directa (si existe)
    cheapest_route: Dict  # Ruta más económica
    direct_exists: bool
    savings: Optional[float] = None  # Ahorro si hay ruta económica mejor


class RouteRequest(BaseModel):
    origin: str
    destination: str
    optimize_by: str = Field(default="cost", description="Criterio: cost o time")
    transport_type: str = Field(default="auto", description="Tipo de transporte: auto, tren, avión o barco")


class RouteResponse(BaseModel):
    """Best path for the chosen criterion, always with both its total cost and hours."""
    origin: str
    destination: str
    optimize_by: str
    path: List[str]
    total_cost: float
    total_hours: float
    toll_cost: Optional[float] = None  # Tolls included in total_cost; None when the transport pays none
    cached: bool = False


class TSPRequest(BaseModel):
    cities: List[str] = Field(..., min_length=2, max_length=MAX_TSP_CITIES)
    cost_matrix: List[List[float]]
    return_to_start: bool = True
    algorithm: Optional[Literal["held_karp", "genetic"]] = None


class TSPResponse(BaseModel):
    """Multi-destination route optimized by the TSP algorithm chosen by the server."""
    algorithm: str
    optimal_route: List[str]
    total_cost: float
    elapsed_ms: float
    history: Optional[List[Dict[str, Any]]] = None
    cached: bool = False


class ItineraryRequest(BaseModel):
    user_id: str
    origin: str
    destinations: List[str]
    max_budget: float = 1000.0
    max_duration_hours: float = 72.0


class ReservationRequest(BaseModel):
    user_id: str
    itinerary: Dict[str, Any]


class ReservationResponse(BaseModel):
    reservation_id: str
    user_id: str
    status: str
    total_cost: float
    created_at: str


# ==========================================================
# GRAFO BASE
# ==========================================================


def get_populated_graph():
    """Devuelve grafo pre-poblado con rutas fijas (auto, tren, avión y barco)."""
    if travel_graph.graph:
        return travel_graph

    for origin, dest, cost, time, transport in ROUTES_FIXED:
        travel_graph.add_route(origin, dest, cost, time, transport, toll=leg_toll(origin, dest, transport))

    logger.info(f"Grafo inicializado con {len(ROUTES_FIXED)} rutas fijas")
    return travel_graph


def mark_as_cached(cached_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a cached result flagged as cached, leaving the stored one intact."""
    result = copy.deepcopy(cached_result)
    result["cached"] = True
    return result


def summarize_path(graph: TravelGraph, path: List[str], transport: str) -> Dict[str, Any]:
    """Return the path with its total cost, hours and tolls (None without tolls), whatever criterion chose it."""
    legs = graph.get_route_details(path, transport)
    return {
        "path": path,
        "total_cost": sum(leg["cost"] for leg in legs),
        "total_hours": round(sum(leg["time"] for leg in legs), HOURS_DECIMALS),
        "toll_cost": sum(leg["toll"] for leg in legs) if transport in TOLLED_TRANSPORTS else None,
    }


# ==========================================================
# ENDPOINTS
# ==========================================================


@app.get("/")
async def root():
    return {
        "message": "Travel Planner API",
        "version": "1.0.0",
        "endpoints": {
            "routes": "/routes/shortest",
            "tsp": "/routes/optimize-multi",
            "matrix": "/routes/matrix",
            "reservations": "/reservations",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_stats": route_cache.get_stats(),
        "reservation_stats": reservation_manager.get_stats()
    }


# ==========================================================
# MATRIZ DE MEJORES RUTAS (DIJKSTRA DESDE CADA CIUDAD)
# ==========================================================


@app.get("/routes/matrix")
async def get_matrix(
    transport: str = "auto",
    optimize_by: str = "cost",
    graph: TravelGraph = Depends(get_populated_graph),
):
    """
    Devuelve, para cada par de ciudades, la mejor ruta según el criterio (con escalas si no hay conexión directa).
    `matrix` trae el valor del criterio; `cost_matrix`, `time_matrix`, `legs_matrix` y `toll_matrix` el costo,
    las horas, los tramos y los peajes de esa misma ruta. Usa -1.0 para rutas no conectadas (infinito),
    ya que JSON no soporta 'inf'.
    """
    if transport not in VALID_TRANSPORTS or optimize_by not in VALID_METRICS:
        raise HTTPException(status_code=400, detail="Parámetros inválidos")

    matrices = await run_in_threadpool(build_path_matrices, graph, transport, optimize_by)
    return {"transport": transport, "optimize_by": optimize_by, **matrices}


def cities_served_by(transport: str) -> List[str]:
    """Return every city with at least one route of the transport, sorted."""
    return sorted({origin for origin, _destination, _cost, _hours, route_transport in ROUTES_FIXED
                   if route_transport == transport})


def totals_matrix(city_totals: List[Dict[str, PathTotals]], cities: List[str], field: str) -> List[List[float]]:
    """Return one field of the path totals between every pair of cities, UNREACHABLE_MARKER when not connected."""
    return [
        [getattr(totals[destination], field) if destination in totals else UNREACHABLE_MARKER
         for destination in cities]
        for totals in city_totals
    ]


# Routes are fixed, so each matrix is computed once per process (one Dijkstra run per city)
@lru_cache(maxsize=len(TRANSPORT_TYPES) * len(VALID_METRICS))
def build_path_matrices(graph: TravelGraph, transport: str, optimize_by: str) -> Dict[str, Any]:
    """Return the criterion, cost, hours, legs and tolls of the best path between every pair of served cities."""
    cities = cities_served_by(transport)
    city_totals = [graph.shortest_path_totals(origin, optimize_by, transport) for origin in cities]
    cost_matrix = totals_matrix(city_totals, cities, "cost")
    time_matrix = [[round(hours, HOURS_DECIMALS) for hours in row]
                   for row in totals_matrix(city_totals, cities, "time")]
    return {
        "cities": cities,
        "matrix": cost_matrix if optimize_by == "cost" else time_matrix,
        "cost_matrix": cost_matrix,
        "time_matrix": time_matrix,
        "legs_matrix": totals_matrix(city_totals, cities, "legs"),
        "toll_matrix": totals_matrix(city_totals, cities, "toll"),
    }


# ==========================================================
# RUTA SIMPLE
# ==========================================================


@app.post("/routes/shortest", response_model=RouteResponse)
async def calculate_shortest_route(request: RouteRequest, graph: TravelGraph = Depends(get_populated_graph)):
    if request.transport_type not in VALID_TRANSPORTS:
        raise HTTPException(status_code=400, detail="Tipo de transporte inválido")

    if request.optimize_by not in VALID_METRICS:
        raise HTTPException(status_code=400, detail="Criterio de optimización inválido")

    validate_known_cities([request.origin, request.destination])

    cache_key = f"{request.origin}_{request.destination}_{request.optimize_by}_{request.transport_type}"

    cached = route_cache.get(cache_key)
    if cached:
        solver_metrics.record_cache_hit()
        return mark_as_cached(cached)

    try:
        started_at = perf_counter()
        path, weight = graph.find_shortest_path(
            request.origin,
            request.destination,
            weight=request.optimize_by,
            transport_type=request.transport_type
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        if not path:
            raise HTTPException(status_code=404, detail="Ruta no encontrada")
        solver_metrics.record_run("dijkstra", SHORTEST_PATH_CITY_COUNT, elapsed_ms, weight, hops=len(path) - 1)

        result = {
            "origin": request.origin,
            "destination": request.destination,
            "optimize_by": request.optimize_by,
            **summarize_path(graph, path, request.transport_type),
            "cached": False,
        }

        route_cache.put(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# COMPARAR RUTAS: DIRECTA vs ECONÓMICA
# ==========================================================

DIRECT_PATH_LENGTH = 2
DIRECT_CONNECTION_DESCRIPTION = "Conexión directa"


def find_direct_route(origin: str, destination: str, transport: str, optimize_by: str) -> Optional[Dict[str, Any]]:
    """Return the direct connection for the transport, or None when it does not exist."""
    for route_origin, route_destination, cost, hours, route_transport in ROUTES_FIXED:
        if (route_origin, route_destination, route_transport) == (origin, destination, transport):
            return {
                "path": [origin, destination],
                "total_cost": cost if optimize_by == "cost" else hours,
                "is_direct": True,
                "description": DIRECT_CONNECTION_DESCRIPTION,
            }
    return None


def describe_cheapest_route(path: List[str], total_cost: float) -> Dict[str, Any]:
    """Describe the Dijkstra route, which may go through intermediate cities."""
    is_direct = len(path) == DIRECT_PATH_LENGTH
    return {
        "path": path,
        "total_cost": total_cost,
        "is_direct": is_direct,
        "description": (
            f"Ruta con {len(path) - 1} segmentos" if len(path) > DIRECT_PATH_LENGTH
            else DIRECT_CONNECTION_DESCRIPTION
        ),
    }


def calculate_savings(direct_route: Optional[Dict[str, Any]], cheapest_cost: float) -> Optional[float]:
    """Return how much the cheapest route saves over the direct one, if anything."""
    if direct_route and cheapest_cost < direct_route["total_cost"]:
        return direct_route["total_cost"] - cheapest_cost
    return None


@app.get("/routes/compare", response_model=RouteComparison)
async def get_compare_routes(
    origin: str,
    destination: str,
    transport: str = "auto",
    optimize_by: str = "cost",
    graph: TravelGraph = Depends(get_populated_graph),
):
    """
    Compara dos opciones de ruta:
    1. Ruta directa (si existe conexión directa en el transporte especificado)
    2. Ruta más económica (usando Dijkstra, puede tener intermediarios)
    """
    if optimize_by not in VALID_METRICS:
        raise HTTPException(status_code=400, detail="Criterio de optimización inválido")

    try:
        direct_route = find_direct_route(origin, destination, transport, optimize_by)
        cheapest_path, cheapest_cost = graph.find_shortest_path(
            origin, destination, weight=optimize_by, transport_type=transport
        )
        if not cheapest_path:
            raise HTTPException(
                status_code=404,
                detail=f"No hay ruta disponible desde {origin} hasta {destination} en {transport}",
            )

        return RouteComparison(
            origin=origin,
            destination=destination,
            direct_route=direct_route,
            cheapest_route=describe_cheapest_route(cheapest_path, cheapest_cost),
            direct_exists=direct_route is not None,
            savings=calculate_savings(direct_route, cheapest_cost),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparando rutas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# COMPARAR TRANSPORTES
# ==========================================================

def best_route_per_transport(
    graph: TravelGraph, origin: str, destination: str, optimize_by: str
) -> List[Dict[str, Any]]:
    """Return the Dijkstra route of every transport able to link both cities, with its total cost and hours."""
    options = []
    for transport in TRANSPORT_TYPES:
        path, _weight = graph.find_shortest_path(origin, destination, weight=optimize_by, transport_type=transport)
        if path:
            options.append({"transport": transport, **summarize_path(graph, path, transport)})
    return options


@app.get("/routes/transports")
async def compare_transports(
    origin: str,
    destination: str,
    optimize_by: str = "cost",
    graph: TravelGraph = Depends(get_populated_graph),
):
    """Compara la mejor ruta de cada transporte entre dos ciudades, marcando la más barata y la más rápida."""
    if optimize_by not in VALID_METRICS:
        raise HTTPException(status_code=400, detail="Criterio de optimización inválido")
    validate_known_cities([origin, destination])

    options = best_route_per_transport(graph, origin, destination, optimize_by)
    if not options:
        raise HTTPException(status_code=404, detail=f"No hay ruta disponible desde {origin} hasta {destination}")

    return {
        "origin": origin,
        "destination": destination,
        "optimize_by": optimize_by,
        "options": options,
        "cheapest": min(options, key=lambda option: option["total_cost"])["transport"],
        "fastest": min(options, key=lambda option: option["total_hours"])["transport"],
    }

# ==========================================================
# TSP MULTIDESTINO
# ==========================================================

GENETIC_POPULATION_SIZE = 200
GENETIC_GENERATIONS = 400
GENETIC_MUTATION_RATE = 0.02
GENETIC_TOURNAMENT_SIZE = 5
GENETIC_ELITISM = 2

NO_FEASIBLE_ROUTE_DETAIL = "No existe una ruta que conecte todas las ciudades con el transporte elegido"


def validate_matrix_matches_cities(cost_matrix: List[List[float]], cities: List[str]) -> None:
    """Raise 422 unless the matrix is square and sized to the city list."""
    n_cities = len(cities)
    if len(cost_matrix) != n_cities or any(len(row) != n_cities for row in cost_matrix):
        raise HTTPException(
            status_code=422,
            detail="La matriz de costos debe ser cuadrada y coincidir con las ciudades",
        )


def validate_known_cities(cities: List[str]) -> None:
    """Raise 422 when any city is not in the catalog."""
    unknown_cities = [city for city in cities if city not in CITIES]
    if unknown_cities:
        raise HTTPException(status_code=422, detail=f"Ciudades desconocidas: {', '.join(unknown_cities)}")


def validate_forced_algorithm(algorithm: Optional[str], n_cities: int) -> None:
    """Raise 422 when Held-Karp is forced beyond its safe size."""
    if algorithm == "held_karp" and n_cities > HELD_KARP_MAX:
        raise HTTPException(status_code=422, detail=f"Held-Karp solo admite hasta {HELD_KARP_MAX} ciudades")


def validate_tsp_request(request: TSPRequest) -> None:
    """Raise 422 when the TSP request is inconsistent or unsafe to solve."""
    validate_matrix_matches_cities(request.cost_matrix, request.cities)
    validate_known_cities(request.cities)
    validate_forced_algorithm(request.algorithm, len(request.cities))


def build_problem_key(request: TSPRequest) -> str:
    """Build a key unique to the cities, tour type and cost matrix of a TSP instance."""
    matrix_hash = hashlib.md5(str(request.cost_matrix).encode()).hexdigest()
    return f"{'-'.join(request.cities)}_{request.return_to_start}_{matrix_hash}"


def build_tsp_cache_key(request: TSPRequest, algorithm: str) -> str:
    """Build a cache key unique to the TSP instance and the algorithm that solves it."""
    return f"multi_{algorithm}_{build_problem_key(request)}"


def replace_unreachable_with_inf(cost_matrix: List[List[float]]) -> List[List[float]]:
    """Replace the JSON-safe unreachable marker with infinity."""
    return [[float("inf") if value == UNREACHABLE_MARKER else value for value in row] for row in cost_matrix]


def solve_with_held_karp(matrix: List[List[float]], cities: List[str], return_to_start: bool) -> Dict[str, Any]:
    """Solve the TSP exactly with Held-Karp."""
    solver = TSPSolver(matrix, cities)
    started_at = perf_counter()
    total_cost, route = solver.solve(start_city=0, return_to_start=return_to_start)
    elapsed_ms = (perf_counter() - started_at) * 1000
    return {
        "algorithm": "held_karp",
        "optimal_route": solver.get_route_with_names(route),
        "total_cost": total_cost,
        "elapsed_ms": elapsed_ms,
        "history": None,
    }


def solve_with_genetic(matrix: List[List[float]], cities: List[str], return_to_start: bool) -> Dict[str, Any]:
    """Solve the TSP approximately with a genetic algorithm."""
    solver = GeneticTSP(
        matrix,
        cities,
        population_size=GENETIC_POPULATION_SIZE,
        generations=GENETIC_GENERATIONS,
        mutation_rate=GENETIC_MUTATION_RATE,
        tournament_size=GENETIC_TOURNAMENT_SIZE,
        elitism=GENETIC_ELITISM,
    )
    total_cost, route = solver.solve(start_city=0, return_to_start=return_to_start)
    return {
        "algorithm": "genetic",
        "optimal_route": solver.get_route_with_names(route),
        "total_cost": total_cost,
        "elapsed_ms": solver.elapsed_ms,
        "history": solver.history,
    }


def tsp_run_quality(algorithm: str, problem_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the genetic convergence and its gap to the exact optimum, when Held-Karp already solved it."""
    if algorithm != "genetic":
        return {}
    return {
        **genetic_convergence(result["history"]),
        "gap_percent": known_optima.gap_percent(problem_key, result["total_cost"]),
    }


def record_tsp_run(request: TSPRequest, algorithm: str, result: Dict[str, Any]) -> None:
    """Record a solved TSP, remembering exact optima so later genetic runs can be compared."""
    problem_key = build_problem_key(request)
    if algorithm == "held_karp":
        known_optima.remember(problem_key, result["total_cost"])
    solver_metrics.record_run(
        algorithm, len(request.cities), result["elapsed_ms"], result["total_cost"],
        **tsp_run_quality(algorithm, problem_key, result),
    )


TSP_SOLVERS = {
    "held_karp": solve_with_held_karp,
    "genetic": solve_with_genetic,
}


def solve_tsp(algorithm: str, matrix: List[List[float]], cities: List[str], return_to_start: bool) -> Dict[str, Any]:
    """Solve the TSP with the named algorithm."""
    return TSP_SOLVERS[algorithm](matrix, cities, return_to_start)


@app.post("/routes/optimize-multi", response_model=TSPResponse)
async def optimize_multi_destination(request: TSPRequest):
    validate_tsp_request(request)
    algorithm = request.algorithm or choose_tsp_algorithm(len(request.cities))
    cache_key = build_tsp_cache_key(request, algorithm)

    cached_result = route_cache.get(cache_key)
    if cached_result:
        solver_metrics.record_cache_hit()
        return mark_as_cached(cached_result)

    try:
        matrix = replace_unreachable_with_inf(request.cost_matrix)
        result = await run_in_threadpool(solve_tsp, algorithm, matrix, request.cities, request.return_to_start)
        if result["total_cost"] == float("inf"):
            raise HTTPException(status_code=422, detail=NO_FEASIBLE_ROUTE_DETAIL)

        result["cached"] = False
        record_tsp_run(request, algorithm, result)
        route_cache.put(cache_key, copy.deepcopy(result))
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en optimize_multi_destination: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ITINERARIO Y RESERVAS
# ==========================================================


def build_itinerary_segments(graph: TravelGraph, origin: str, destinations: List[str]) -> List[Dict[str, Any]]:
    """Chain shortest paths through the destinations, skipping unreachable ones."""
    segments = []
    current = origin
    for destination in destinations:
        path, cost = graph.find_shortest_path(current, destination)
        if path:
            segments.append({"from": current, "to": destination, "path": path, "cost": cost})
            current = destination
    return segments


@app.post("/itinerary/plan")
async def plan_itinerary(request: ItineraryRequest, graph: TravelGraph = Depends(get_populated_graph)):
    try:
        segments = build_itinerary_segments(graph, request.origin, request.destinations)
        total_cost = sum(segment["cost"] for segment in segments)
        within_budget = total_cost <= request.max_budget
        return {
            "user_id": request.user_id,
            "origin": request.origin,
            "destinations": request.destinations,
            "segments": segments,
            "total_cost": total_cost,
            "within_budget": within_budget,
            "valid": within_budget,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# RESERVAS (individuales y por lote)
# ==========================================================


@app.post("/reservations", response_model=ReservationResponse)
async def create_reservation(request: ReservationRequest, background_tasks: BackgroundTasks):
    """Crea una NUEVA reserva individual (procesamiento inmediato)."""
    try:
        reservation = await reservation_manager.create_reservation(
            user_id=request.user_id,
            itinerary=request.itinerary
        )

        async def process_async(reservation_obj):
            await reservation_manager.process_reservation(reservation_obj)
            logger.info(f"Task: Reserva individual {reservation_obj.reservation_id} finalizada")

        background_tasks.add_task(process_async, reservation)

        logger.info(f"Reserva individual {reservation.reservation_id} creada y encolada.")
        return reservation.to_dict()
    except Exception as e:
        logger.error(f"Error creando reserva: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reservations/batch")
async def create_reservations_batch(requests: List[ReservationRequest]):
    """
    Añade múltiples reservas al PROCESADOR POR LOTES (BatchProcessor).
    """
    if not requests:
        raise HTTPException(status_code=400, detail="La lista de reservas no puede estar vacía")

    logger.info(f"Recibido lote de {len(requests)} reservas para User {requests[0].user_id}")

    try:
        for reservation_request in requests:
            batch_processor.add_item_sync(item_id=reservation_request.user_id, data=reservation_request.itinerary)

        return {
            "status": "queued",
            "count": len(requests),
            "message": f"{len(requests)} reservas añadidas al lote. Se procesarán en breve."
        }

    except Exception as e:
        logger.error(f"Error creando lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reservations/{reservation_id}")
async def get_reservation(reservation_id: str):
    reservation = reservation_manager.get_reservation(reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail="Reserva no encontrada")
    return reservation.to_dict()


@app.get("/reservations/user/{user_id}")
async def get_user_reservations(user_id: str):
    reservations = reservation_manager.get_user_reservations(user_id)
    return [r.to_dict() for r in reservations]


@app.delete("/reservations/{reservation_id}")
async def cancel_reservation(reservation_id: str):
    success = await reservation_manager.cancel_reservation(reservation_id)
    if not success:
        raise HTTPException(status_code=400, detail="No se pudo cancelar la reserva")
    return {"message": "Reserva cancelada exitosamente"}


# ==========================================================
# ESTADÍSTICAS DEL SISTEMA
# ==========================================================


@app.get("/stats")
async def get_system_stats():
    return {
        "cache": route_cache.get_stats(),
        "reservations": reservation_manager.get_stats(),
        "batch_processor": batch_processor.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats/algorithms")
async def get_algorithm_stats():
    """Return the real solver runs and their aggregates per algorithm and city count."""
    return solver_metrics.snapshot()


# ==========================================================
# MAIN (para ejecutar localmente)
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
