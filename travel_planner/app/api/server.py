"""
server.py - Servidor FastAPI que integra todos los módulos del sistema.
API RESTful para el sistema de planificación de viajes multidestino.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
from app.data.routes_fixed import ROUTES_FIXED
import asyncio
import logging

# Imports de nuestros módulos
from app.core.graph import TravelGraph
from app.core.tsp_dp import TSPSolver
from app.core.itinerary_validator import ItineraryConstraints
from app.caches.lru_cache import LRUCache
from app.booking.reservations import ReservationManager
from app.booking.batching import ReservationBatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN PRINCIPAL
# ==========================================================

app = FastAPI(
    title="Travel Planner API",
    description="Sistema de planificación de viajes multidestino con optimización algorítmica",
    version="1.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias globales
travel_graph = TravelGraph()
route_cache = LRUCache(capacity=100)
reservation_manager = ReservationManager(max_concurrent=10)
batch_processor = ReservationBatchProcessor(batch_size=20)

# ==========================================================
# MODELOS Pydantic
# ==========================================================

class RouteRequest(BaseModel):
    origin: str
    destination: str
    optimize_by: str = Field(default="cost", description="Criterio: cost o time")

class RouteResponse(BaseModel):
    origin: str
    destination: str
    path: List[str]
    total_cost: float
    cached: bool = False

class TSPRequest(BaseModel):
    cities: List[str] = Field(..., min_length=2)
    cost_matrix: List[List[float]]
    return_to_start: bool = True

class TSPResponse(BaseModel):
    optimal_route: List[str]
    total_cost: float
    computation_time: float

class ItineraryRequest(BaseModel):
    user_id: str
    origin: str
    destinations: List[str]
    max_budget: float = 1000.0
    max_duration_hours: float = 72.0
    transport_preferences: List[str] = ["tren", "avión", "bus"]

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
    """Devuelve grafo pre-poblado con rutas fijas (auto, avión, tren)."""
    if travel_graph.graph:
        return travel_graph

    for origin, dest, cost, time, transport in ROUTES_FIXED:
        travel_graph.add_route(origin, dest, cost, time, transport)

    logger.info(f"Grafo inicializado con {len(ROUTES_FIXED)} rutas fijas")
    return travel_graph

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
# 🔹 NUEVO ENDPOINT: MATRIZ DESDE ROUTES_FIXED
# ==========================================================

@app.get("/routes/matrix")
async def get_matrix(transport: str = "auto", optimize_by: str = "cost"):
    """
    Devuelve matriz de costos o tiempos fijos desde app/data/routes_fixed.py
    """
    valid_transports = {"auto", "avión", "tren"}
    valid_metrics = {"cost", "time"}

    if transport not in valid_transports or optimize_by not in valid_metrics:
        raise HTTPException(status_code=400, detail="Parámetros inválidos")

    filtered = [r for r in ROUTES_FIXED if r[4] == transport]
    cities = sorted({r[0] for r in filtered} | {r[1] for r in filtered})
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]

    for (o, d, cost, time, t) in filtered:
        i, j = cities.index(o), cities.index(d)
        matrix[i][j] = cost if optimize_by == "cost" else time
        matrix[j][i] = matrix[i][j]

    return {
        "cities": cities,
        "transport": transport,
        "optimize_by": optimize_by,
        "matrix": matrix
    }

# ==========================================================
# RUTA SIMPLE
# ==========================================================

@app.post("/routes/shortest", response_model=RouteResponse)
async def calculate_shortest_route(request: RouteRequest, graph: TravelGraph = Depends(get_populated_graph)):
    cache_key = f"{request.origin}_{request.destination}_{request.optimize_by}"

    cached = route_cache.get(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        path, cost = graph.find_shortest_path(
            request.origin,
            request.destination,
            weight=request.optimize_by
        )
        if not path:
            raise HTTPException(status_code=404, detail="Ruta no encontrada")

        result = {
            "origin": request.origin,
            "destination": request.destination,
            "path": path,
            "total_cost": cost,
            "cached": False
        }
        route_cache.put(cache_key, result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# RUTA MULTIDESTINO (TSP) con Cache
# ==========================================================

@app.post("/routes/optimize-multi", response_model=TSPResponse)
async def optimize_multi_destination(request: TSPRequest):
    """
    Optimiza una ruta multidestino usando TSP con memoización (cache LRU).
    Si ya existe una combinación idéntica de ciudades y parámetros, se devuelve desde cache.
    """
    import time

    start = time.time()

    # Crear clave de cache única
    # Ejemplo: "multi_Madrid-Barcelona-París_auto_cost_return"
    cache_key = f"multi_{'-'.join(request.cities)}_{request.return_to_start}"

    # 1️⃣ Buscar en cache
    cached_result = route_cache.get(cache_key)
    if cached_result:
        logger.info(f"🧠 Resultado obtenido desde cache: {cache_key}")
        return {**cached_result, "cached": True}

    try:
        # 2️⃣ Resolver TSP normalmente
        solver = TSPSolver(cost_matrix=request.cost_matrix, city_names=request.cities)
        min_cost, route_idx = solver.solve(start_city=0, return_to_start=request.return_to_start)
        route = solver.get_route_with_names(route_idx)
        elapsed = time.time() - start

        # 3️⃣ Armar resultado
        result = {
            "optimal_route": route,
            "total_cost": min_cost,
            "computation_time": elapsed,
            "cached": False
        }

        # 4️⃣ Guardar en cache
        route_cache.put(cache_key, result)
        logger.info(f"💾 Guardado en cache: {cache_key}")

        return result

    except Exception as e:
        logger.error(f"❌ Error en optimize_multi_destination: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# ITINERARIO Y RESERVAS
# ==========================================================

@app.post("/itinerary/plan")
async def plan_itinerary(request: ItineraryRequest):
    try:
        segments = []
        current = request.origin
        for destination in request.destinations:
            path, cost = travel_graph.find_shortest_path(current, destination)
            if path:
                segments.append({"from": current, "to": destination, "path": path, "cost": cost})
                current = destination

        constraints = ItineraryConstraints(
            max_budget=request.max_budget,
            max_duration_hours=request.max_duration_hours,
            max_segments=10,
            required_cities=request.destinations
        )

        total_cost = sum(s["cost"] for s in segments)
        return {
            "user_id": request.user_id,
            "origin": request.origin,
            "destinations": request.destinations,
            "segments": segments,
            "total_cost": total_cost,
            "within_budget": total_cost <= request.max_budget,
            "valid": total_cost <= request.max_budget
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 🧩 RESERVAS (individuales y por lote)
# ==========================================================

@app.post("/reservations", response_model=ReservationResponse)
async def create_reservation(request: ReservationRequest, background_tasks: BackgroundTasks):
    """Crea una nueva reserva individual."""
    try:
        reservation = await reservation_manager.create_reservation(
            user_id=request.user_id,
            itinerary=request.itinerary
        )

        async def process_async(reservation_obj):
            await reservation_manager.process_reservation(reservation_obj)

        background_tasks.add_task(process_async, reservation)
        logger.info(f"Reserva creada correctamente: {reservation.reservation_id}")

        return {
            "reservation_id": reservation.reservation_id,
            "user_id": reservation.user_id,
            "status": reservation.status.value,
            "total_cost": reservation.total_cost,
            "created_at": reservation.created_at.isoformat()
        }

    except Exception as e:
        logger.error(f"Error creando reserva: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reservations/batch")
async def create_reservations_batch(requests: List[ReservationRequest], background_tasks: BackgroundTasks):
    """
    Crea múltiples reservas y las procesa en batch automáticamente.
    Cada reserva se procesa asíncronamente igual que en create_reservation(),
    garantizando que todas pasen de pending → confirmed de forma normal.
    """
    created_ids = []

    try:
        for req in requests:
            # Crear la reserva base (queda inicialmente en estado pending)
            reservation = await reservation_manager.create_reservation(
                user_id=req.user_id,
                itinerary=req.itinerary
            )
            created_ids.append(reservation.reservation_id)

            # Procesarla de forma asíncrona igual que las individuales
            async def process_async(reservation_obj):
                await reservation_manager.process_reservation(reservation_obj)

            background_tasks.add_task(process_async, reservation)

        logger.info(f"🧩 Lote recibido con {len(requests)} reservas para procesamiento asíncrono")

        return {
            "status": "queued",
            "count": len(requests),
            "reservations": created_ids,
            "message": f"{len(requests)} reservas creadas y en proceso de confirmación."
        }

    except Exception as e:
        logger.error(f"Error creando lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

##
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

# 💡 Procesamiento automático de batches cada X segundos
@app.on_event("startup")
async def start_batch_loop():
    """Ejecuta procesamiento periódico de lotes."""
    async def loop():
        while True:
            if batch_processor.queue:
                logger.info(f"⏳ Procesando lote automático ({len(batch_processor.queue)} en cola)")
                batch = batch_processor._extract_batch()
                if batch:
                    await batch_processor._process_batch(batch)
            await asyncio.sleep(3)

    asyncio.create_task(loop())





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
