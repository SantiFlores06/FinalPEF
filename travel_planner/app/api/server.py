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
import time # Importar time para el profiling

# Imports de nuestros módulos
from app.core.graph import TravelGraph
from app.core.tsp_dp import TSPSolver
from app.core.itinerary_validator import ItineraryConstraints
from app.caches.lru_cache import LRUCache
from app.booking.reservations import ReservationManager
from app.booking.batching import ReservationBatchProcessor
from app.ml.data_loader import DataLoader
from app.ml.recommender import TravelRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INICIAR IA AL ARRANCAR ---
logger.info("Cargando datos de IA...")
data_loader = DataLoader(data_dir="app/data")
recommender = TravelRecommender(data_loader=data_loader)
recommender.train_knn_model()
logger.info("Sistema de Recomendaciones de IA listo.")
# -----------------------------

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
batch_processor = ReservationBatchProcessor(
    batch_size=20, 
    timeout_seconds=5.0, # (El timeout que tenías en batching.py)
    reservation_manager=reservation_manager 
)

# ==========================================================
# MODELOS Pydantic (Corregidos)
# ==========================================================
class AIRecommendation(BaseModel):
    """Modelo para una sola recomendación de IA"""
    destination_id: str
    destination_name: str
    similarity: float

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
    recommendations: List[AIRecommendation] = [] # Ya estaba aquí

class TSPRequest(BaseModel):
    cities: List[str] = Field(..., min_length=2)
    cost_matrix: List[List[float]]
    return_to_start: bool = True

class TSPResponse(BaseModel):
    """Modelo TSP Corregido"""
    optimal_route: List[str]
    total_cost: float
    computation_time: float
    cached: bool = False                 # <-- AÑADIDO
    recommendations: List[AIRecommendation] = [] # <-- AÑADIDO

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
    if travel_graph.graph: # Revisa si el grafo ya tiene nodos/aristas
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
    
    # Inicializar con infinito para destinos no conectados
    matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0.0 # Costo 0 a sí mismo

    for (o, d, cost, time, t) in filtered:
        if o in cities and d in cities: # Asegurarse que ambas ciudades estén en la lista
            i, j = cities.index(o), cities.index(d)
            value = cost if optimize_by == "cost" else time
            matrix[i][j] = value
            matrix[j][i] = value # Asumir rutas simétricas para TSP
    # Convertir el Inf en None para que JSON sea compatible
    json_compliant_matrix = [
        [None if val == float('inf') else val for val in row]
        for row in matrix
    ]
    return {
        "cities": cities,
        "transport": transport,
        "optimize_by": optimize_by,
        "matrix": json_compliant_matrix # <-- Enviar la matriz corregida
    }

# ==========================================================
# RUTA SIMPLE
# ==========================================================

@app.post("/routes/shortest", response_model=RouteResponse)
async def calculate_shortest_route(request: RouteRequest, graph: TravelGraph = Depends(get_populated_graph)):
    cache_key = f"{request.origin}_{request.destination}_{request.optimize_by}"

    cached = route_cache.get(cache_key)
    if cached:
        cached["cached"] = True
        return cached # Devuelve el resultado completo desde el caché

    try:
        path, cost = graph.find_shortest_path(
            request.origin,
            request.destination,
            weight=request.optimize_by
        )
        if not path:
            raise HTTPException(status_code=404, detail="Ruta no encontrada")

        # --- INICIO DE INTEGRACIÓN CON IA ---
        ai_recs = []
        final_destination = path[-1] # Obtener el destino final de la ruta
        dest_id = final_destination.lower() # Convertir a ID (ej. "París" -> "paris")

        if dest_id in data_loader.destinations:
            # Llamar al modelo KNN para encontrar destinos similares
            similar_destinations = recommender.get_similar_destinations(
                destination_id=dest_id, 
                n_similar=3
            )
            # Formatear la respuesta de la IA
            for dest_key, sim_score in similar_destinations:
                dest_obj = data_loader.get_destination(dest_key)
                if dest_obj:
                    ai_recs.append({
                        "destination_id": dest_obj.id,
                        "destination_name": dest_obj.name,
                        "similarity": sim_score
                    })
        else:
            logger.warning(f"Destino '{dest_id}' no encontrado en los datos de la IA.")
        # --- FIN DE INTEGRACIÓN CON IA ---

        result = {
            "origin": request.origin,
            "destination": request.destination,
            "path": path,
            "total_cost": cost,
            "cached": False,
            "recommendations": ai_recs  # <-- Añadir recomendaciones al resultado
        }
        
        route_cache.put(cache_key, result) # Guardar el resultado completo en caché
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# RUTA MULTIDESTINO (TSP) con Cache (Corregido)
# ==========================================================

@app.post("/routes/optimize-multi", response_model=TSPResponse)
async def optimize_multi_destination(request: TSPRequest):
    """
    Optimiza una ruta multidestino usando TSP con memoización (cache LRU).
    Si ya existe una combinación idéntica de ciudades y parámetros, se devuelve desde cache.
    """
    start = time.time()

    # Crear clave de cache única
    cache_key = f"multi_{'-'.join(request.cities)}_{request.return_to_start}"

    # 1️⃣ Buscar en cache
    cached_result = route_cache.get(cache_key)
    if cached_result:
        logger.info(f"🧠 Resultado obtenido desde cache: {cache_key}")
        cached_result["cached"] = True # Asegúrate de marcarlo como cacheado
        return cached_result

    try:
        # 2️⃣ Resolver TSP normalmente
        solver = TSPSolver(cost_matrix=request.cost_matrix, city_names=request.cities)
        min_cost, route_idx = solver.solve(start_city=0, return_to_start=request.return_to_start)
        route = solver.get_route_with_names(route_idx)
        elapsed = time.time() - start

        # --- INICIO DE INTEGRACIÓN CON IA ---
        ai_recs = []
        final_destination = route[-1] # Destino final de la ruta TSP
        dest_id = final_destination.lower() 

        if dest_id in data_loader.destinations:
            # Llamar al modelo KNN
            similar_destinations = recommender.get_similar_destinations(
                destination=dest_id, 
                n_similar=3
            )
            # Formatear la respuesta
            for dest_key, sim_score in similar_destinations:
                dest_obj = data_loader.get_destination(dest_key)
                if dest_obj:
                    ai_recs.append({
                        "destination_id": dest_obj.id,
                        "destination_name": dest_obj.name,
                        "similarity": sim_score
                    })
        else:
            logger.warning(f"Destino '{dest_id}' no encontrado en los datos de la IA.")
        # --- FIN DE INTEGRACIÓN CON IA ---

        #  Armar resultado
        result = {
            "optimal_route": route,
            "total_cost": min_cost,
            "computation_time": elapsed,
            "recommendations": ai_recs, # <-- IA AÑADIDA
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
async def plan_itinerary(request: ItineraryRequest, graph: TravelGraph = Depends(get_populated_graph)):
    try:
        segments = []
        current = request.origin
        for destination in request.destinations:
            path, cost = graph.find_shortest_path(current, destination)
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
            "valid": total_cost <= request.max_budget # Lógica de validación simplificada
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 🧩 RESERVAS (individuales y por lote) - CORREGIDO
# ==========================================================

@app.post("/reservations", response_model=ReservationResponse)
async def create_reservation(request: ReservationRequest, background_tasks: BackgroundTasks):
    """
    Crea una NUEVA reserva individual (procesamiento inmediato).
    """
    try:
        # 1. Crear la reserva en el manager (estado: PENDING)
        reservation = await reservation_manager.create_reservation(
            user_id=request.user_id,
            itinerary=request.itinerary
        )

        # 2. Definir la tarea asíncrona que se ejecutará en fondo
        async def process_async(reservation_obj):
            logger.info(f"Task: Procesando reserva individual {reservation_obj.reservation_id}")
            await reservation_manager.process_reservation(reservation_obj)
            logger.info(f"Task: Reserva individual {reservation_obj.reservation_id} finalizada")

        # 3. Añadir la tarea al fondo de FastAPI
        background_tasks.add_task(process_async, reservation)
        
        logger.info(f"Reserva individual {reservation.reservation_id} creada y encolada.")

        # 4. Devolver respuesta inmediata al usuario
        return {
            "reservation_id": reservation.reservation_id,
            "user_id": reservation.user_id,
            "status": reservation.status.value, # Devolverá "pending"
            "total_cost": reservation.total_cost,
            "created_at": reservation.created_at.isoformat()
        }

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

    user_id = requests[0].user_id
    
    logger.info(f"🧩 Recibido lote de {len(requests)} reservas para User {user_id}")

    try:
        # 1. Añadir todos los items a la cola RÁPIDAMENTE
        #    (Ahora usamos add_item_sync y no hay 'await')
        for req in requests:
            batch_processor.add_item_sync(item_id=req.user_id, data=req.itinerary) 

        # 2. Devolver respuesta inmediata
        #    El servidor responde "En cola" en menos de 1 segundo.
        return {
            "status": "queued",
            "count": len(requests),
            "message": f"{len(requests)} reservas añadidas al lote. Se procesarán en breve."
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
    logger.info("⏰ Iniciando loop de procesamiento de lotes en background...")
    
    async def loop():
        while True:
            await asyncio.sleep(10) # Revisa cada 10 segundos
            
            # CORRECCIÓN: Llamar a la función que SÍ existe en tu batching.py
            await batch_processor._trigger_processing()

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