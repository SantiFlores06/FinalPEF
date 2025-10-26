"""
server.py - Servidor FastAPI que integra todos los módulos del sistema.
API RESTful para el sistema de planificación de viajes multidestino.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Imports de nuestros módulos
from app.core.graph import TravelGraph
from app.core.tsp_dp import TSPSolver
from app.core.itinerary_validator import (
    ItineraryValidator,
    ItineraryConstraints,
    RouteSegment,
    TransportType
)
from app.caches.lru_cache import LRUCache
from app.booking.reservations import ReservationManager, Reservation
from app.booking.batching import ReservationBatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Travel Planner API",
    description="Sistema de planificación de viajes multidestino con optimización algorítmica",
    version="1.0.0"
)

# Configurar CORS
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

# Modelos Pydantic para requests/responses
class RouteRequest(BaseModel):
    """Request para calcular una ruta."""
    origin: str = Field(..., example="Madrid")
    destination: str = Field(..., example="Barcelona")
    optimize_by: str = Field(default="cost", example="cost")

class RouteResponse(BaseModel):
    """Response con la ruta calculada."""
    origin: str
    destination: str
    path: List[str]
    total_cost: float
    cached: bool = False

class TSPRequest(BaseModel):
    """Request para resolver TSP."""
    cities: List[str] = Field(..., min_length=2, example=["Madrid", "Barcelona", "Valencia"])
    cost_matrix: List[List[float]]
    return_to_start: bool = Field(default=True)

class TSPResponse(BaseModel):
    """Response con solución TSP."""
    optimal_route: List[str]
    total_cost: float
    computation_time: float

class ItineraryRequest(BaseModel):
    """Request para crear itinerario completo."""
    user_id: str
    origin: str
    destinations: List[str]
    max_budget: float = Field(default=1000.0)
    max_duration_hours: float = Field(default=72.0)
    transport_preferences: List[str] = Field(default=["tren", "avión", "bus"])

class ReservationRequest(BaseModel):
    """Request para crear reserva."""
    user_id: str
    itinerary: Dict[str, Any]

class ReservationResponse(BaseModel):
    """Response con información de reserva."""
    reservation_id: str
    user_id: str
    status: str
    total_cost: float
    created_at: str

# Dependency para inicializar grafo con datos de ejemplo
def get_populated_graph():
    """Devuelve grafo pre-poblado con rutas de ejemplo."""
    if travel_graph.graph:
        return travel_graph
    
    # Datos de ejemplo - en producción vendría de base de datos
    routes = [
        ("Madrid", "Barcelona", 50, 3, "tren"),
        ("Madrid", "Valencia", 40, 4, "bus"),
        ("Madrid", "Sevilla", 60, 5, "tren"),
        ("Barcelona", "París", 100, 2, "avión"),
        ("Barcelona", "Valencia", 45, 3.5, "tren"),
        ("Valencia", "París", 120, 8, "bus"),
        ("París", "Roma", 150, 2.5, "avión"),
        ("Roma", "Barcelona", 130, 2, "avión"),
        ("Sevilla", "Lisboa", 50, 6, "bus"),
        ("Lisboa", "Madrid", 55, 6.5, "bus"),
    ]
    
    for origin, dest, cost, time, transport in routes:
        travel_graph.add_route(origin, dest, cost, time, transport)
    
    logger.info(f"Grafo inicializado con {len(routes)} rutas")
    return travel_graph

# Endpoints

@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Travel Planner API",
        "version": "1.0.0",
        "endpoints": {
            "routes": "/routes/shortest",
            "tsp": "/routes/optimize-multi",
            "reservations": "/reservations",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_stats": route_cache.get_stats(),
        "reservation_stats": reservation_manager.get_stats()
    }

@app.post("/routes/shortest", response_model=RouteResponse)
async def calculate_shortest_route(
    request: RouteRequest,
    graph: TravelGraph = Depends(get_populated_graph)
):
    """
    Calcula la ruta más corta entre dos ciudades usando Dijkstra.
    
    - **origin**: Ciudad de origen
    - **destination**: Ciudad de destino
    - **optimize_by**: Criterio de optimización (cost o time)
    """
    cache_key = f"{request.origin}_{request.destination}_{request.optimize_by}"
    
    # Intentar obtener del cache
    cached_result = route_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache HIT para ruta {cache_key}")
        return {**cached_result, "cached": True}
    
    # Calcular ruta
    try:
        path, cost = graph.find_shortest_path(
            request.origin,
            request.destination,
            weight=request.optimize_by
        )
        
        if not path:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ruta entre {request.origin} y {request.destination}"
            )
        
        result = {
            "origin": request.origin,
            "destination": request.destination,
            "path": path,
            "total_cost": cost,
            "cached": False
        }
        
        # Guardar en cache
        route_cache.put(cache_key, result)
        logger.info(f"Ruta calculada y cacheada: {cache_key}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculando ruta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/routes/optimize-multi", response_model=TSPResponse)
async def optimize_multi_destination(request: TSPRequest):
    """
    Optimiza ruta visitando múltiples destinos (TSP).
    
    Usa programación dinámica para encontrar el orden óptimo de visita.
    """
    import time
    start_time = time.time()
    
    try:
        solver = TSPSolver(
            cost_matrix=request.cost_matrix,
            city_names=request.cities
        )
        
        min_cost, route_indices = solver.solve(
            start_city=0,
            return_to_start=request.return_to_start
        )
        
        optimal_route = solver.get_route_with_names(route_indices)
        computation_time = time.time() - start_time
        
        logger.info(f"TSP resuelto en {computation_time:.3f}s para {len(request.cities)} ciudades")
        
        return {
            "optimal_route": optimal_route,
            "total_cost": min_cost,
            "computation_time": computation_time
        }
        
    except Exception as e:
        logger.error(f"Error en TSP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/itinerary/plan")
async def plan_itinerary(request: ItineraryRequest):
    """
    Planifica itinerario completo considerando múltiples restricciones.
    
    Combina Dijkstra para rutas individuales y validación de restricciones.
    """
    try:
        # Calcular rutas entre destinos consecutivos
        segments = []
        current = request.origin
        
        for destination in request.destinations:
            path, cost = travel_graph.find_shortest_path(current, destination)
            if path:
                segments.append({
                    "from": current,
                    "to": destination,
                    "path": path,
                    "cost": cost
                })
                current = destination
        
        # Validar itinerario
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
        logger.error(f"Error planificando itinerario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reservations", response_model=ReservationResponse)
async def create_reservation(
    request: ReservationRequest,
    background_tasks: BackgroundTasks
):
    """
    Crea una nueva reserva de viaje.
    
    La reserva se procesa de forma asíncrona en background.
    """
    try:
        reservation = await reservation_manager.create_reservation(
            user_id=request.user_id,
            itinerary=request.itinerary
        )
        
        # Procesar en background
        background_tasks.add_task(
            reservation_manager.process_reservation,
            reservation
        )
        
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

@app.get("/reservations/{reservation_id}")
async def get_reservation(reservation_id: str):
    """Obtiene el estado de una reserva."""
    reservation = reservation_manager.get_reservation(reservation_id)
    
    if not reservation:
        raise HTTPException(status_code=404, detail="Reserva no encontrada")
    
    return reservation.to_dict()

@app.get("/reservations/user/{user_id}")
async def get_user_reservations(user_id: str):
    """Obtiene todas las reservas de un usuario."""
    reservations = reservation_manager.get_user_reservations(user_id)
    return [r.to_dict() for r in reservations]

@app.delete("/reservations/{reservation_id}")
async def cancel_reservation(reservation_id: str):
    """Cancela una reserva."""
    success = await reservation_manager.cancel_reservation(reservation_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="No se pudo cancelar la reserva"
        )
    
    return {"message": "Reserva cancelada exitosamente"}

@app.get("/stats")
async def get_system_stats():
    """Obtiene estadísticas del sistema."""
    return {
        "cache": route_cache.get_stats(),
        "reservations": reservation_manager.get_stats(),
        "batch_processor": batch_processor.get_stats(),
        "timestamp": datetime.now().isoformat()
    }

# Ejecutar servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
