"""
test_graph.py - Tests unitarios para el algoritmo de Dijkstra.
"""
import pytest
from app.core.graph import TravelGraph

def test_add_route(simple_graph):
    """Verifica que las rutas se agreguen correctamente al grafo."""
    # El grafo debe tener 3 vértices: A, B, C
    assert len(simple_graph.vertices) == 3
    # A debe tener 2 vecinos (B y C)
    assert len(simple_graph.graph["A"]) == 2

def test_dijkstra_shortest_path_cost(simple_graph):
    """
    Prueba Dijkstra optimizando por COSTO.
    Camino esperado: A -> B -> C (Costo 10 + 20 = 30)
    La directa A -> C cuesta 50 (es peor).
    """
    path, cost = simple_graph.find_shortest_path("A", "C", weight="cost")
    
    assert path == ["A", "B", "C"]
    assert cost == 30

def test_dijkstra_shortest_path_time(simple_graph):
    """
    Prueba Dijkstra optimizando por TIEMPO.
    Camino esperado: A -> C Directo (Tiempo 0.5)
    A -> B -> C tarda 1 + 2 = 3 (es peor).
    """
    path, time = simple_graph.find_shortest_path("A", "C", weight="time")
    
    assert path == ["A", "C"]
    assert time == 0.5

def test_no_path():
    """Verifica que devuelva lista vacía e infinito si no hay camino."""
    graph = TravelGraph()
    graph.add_route("A", "B", 10, 1, "bus")
    graph.add_vertex("Z") # Isla aislada
    
    path, cost = graph.find_shortest_path("A", "Z")
    
    assert path == []
    assert cost == float('inf')

def test_unknown_node():
    """Verifica el manejo de nodos que no existen."""
    graph = TravelGraph()
    graph.add_route("A", "B", 10, 1, "bus")
    
    path, cost = graph.find_shortest_path("A", "X")
    assert path == []