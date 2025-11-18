"""
profiling_analysis.py - Script para análisis de rendimiento (CPU y Memoria).
Mide el desempeño de los algoritmos Core (Dijkstra y TSP).
"""
import sys
import os
import time
import cProfile
import pstats
import random
import io
from memory_profiler import profile as memory_profile

# Asegurar que podemos importar los módulos de la app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.graph import TravelGraph
from app.core.tsp_dp import TSPSolver

# ==========================================
# CONFIGURACIÓN DE PRUEBA
# ==========================================
NUM_CITIES_DIJKSTRA = 1000
NUM_ROUTES_DIJKSTRA = 5000
NUM_CITIES_TSP = 18  # Cuidado: TSP es exponencial. >20 tardará mucho.

# ==========================================
# 1. PROFILING DE DIJKSTRA (Grafo Grande)
# ==========================================

@memory_profile
def test_dijkstra_performance():
    print(f"\n--- 🧪 Iniciando Test de Estrés Dijkstra ({NUM_CITIES_DIJKSTRA} ciudades) ---")
    
    # 1. Construir un grafo grande aleatorio
    graph = TravelGraph()
    cities = [f"City_{i}" for i in range(NUM_CITIES_DIJKSTRA)]
    
    print(f"  Generando {NUM_ROUTES_DIJKSTRA} rutas aleatorias...")
    for _ in range(NUM_ROUTES_DIJKSTRA):
        origin = random.choice(cities)
        dest = random.choice(cities)
        if origin != dest:
            graph.add_route(
                origin=origin,
                destination=dest,
                cost=random.uniform(10, 500),
                time=random.uniform(1, 10),
                transport_type="auto"
            )
    
    # 2. Ejecutar búsquedas
    print("  Ejecutando 100 búsquedas de ruta mínima...")
    start_time = time.time()
    
    found = 0
    for _ in range(100):
        o = random.choice(cities)
        d = random.choice(cities)
        path, cost = graph.find_shortest_path(o, d)
        if path:
            found += 1
            
    elapsed = time.time() - start_time
    print(f"  ✅ Finalizado en {elapsed:.4f} segundos.")
    print(f"  Rutas encontradas: {found}/100")

# ==========================================
# 2. PROFILING DE TSP (Carga Exponencial)
# ==========================================

@memory_profile
def test_tsp_performance():
    print(f"\n--- 🧪 Iniciando Test de Estrés TSP ({NUM_CITIES_TSP} ciudades) ---")
    print("  Advertencia: Esto usa O(n^2 * 2^n), puede tardar unos segundos.")
    
    # 1. Crear matriz de costos aleatoria
    matrix = [
        [0 if i == j else random.uniform(10, 100) for j in range(NUM_CITIES_TSP)]
        for i in range(NUM_CITIES_TSP)
    ]
    
    city_names = [f"C{i}" for i in range(NUM_CITIES_TSP)]
    
    # 2. Resolver TSP
    print("  Resolviendo TSP...")
    start_time = time.time()
    
    solver = TSPSolver(cost_matrix=matrix, city_names=city_names)
    cost, route = solver.solve(start_city=0)
    
    elapsed = time.time() - start_time
    print(f"  ✅ Finalizado en {elapsed:.4f} segundos.")
    print(f"  Costo mínimo: {cost:.2f}")
    print(f"  Ruta calculada: {len(route)} pasos.")

# ==========================================
# MAIN CON CPROFILE
# ==========================================

if __name__ == "__main__":
    print("========================================")
    print(" EJECUTANDO ANÁLISIS DE PERFILADO")
    print("========================================")
    
    # Usamos cProfile para medir tiempo de CPU detallado
    profiler = cProfile.Profile()
    profiler.enable()
    
    test_dijkstra_performance()
    test_tsp_performance()
    
    profiler.disable()
    
    print("\n========================================")
    print(" REPORTE DE CPU (TOP 10 FUNCIONES)")
    print("========================================")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10) # Mostrar las 10 funciones más lentas
    print(s.getvalue())
    
    print("Análisis completado.")