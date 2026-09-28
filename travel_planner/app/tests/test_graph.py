"""
test_graph.py - Tests unitarios para el algoritmo de Dijkstra.
"""
import pytest
from app.core.graph import PathTotals, TravelGraph

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

def test_path_totals_sum_legs_of_the_requested_transport():
    graph = TravelGraph()
    graph.add_route("A", "B", 10, 1, "tren")
    graph.add_route("A", "B", 99, 9, "avión")
    graph.add_route("B", "C", 20, 2.5, "tren")

    assert graph.path_totals(["A", "B", "C"], "tren") == (30, 3.5)

def test_shortest_path_totals_follow_the_cheapest_paths(simple_graph):
    totals = simple_graph.shortest_path_totals("A", weight="cost")

    assert totals == {
        "A": PathTotals(),
        "B": PathTotals(cost=10, time=1, legs=1),
        "C": PathTotals(cost=30, time=3, legs=2),
    }


def test_shortest_path_totals_follow_the_fastest_paths(simple_graph):
    totals = simple_graph.shortest_path_totals("A", weight="time")

    assert totals["C"] == PathTotals(cost=50, time=0.5, legs=1)


def test_shortest_path_totals_skip_unreachable_cities_and_other_transports(simple_graph):
    simple_graph.add_vertex("Z")

    totals = simple_graph.shortest_path_totals("A", transport_type="bus")

    assert set(totals) == {"A", "B"}


def test_shortest_path_totals_add_up_the_tolls_of_every_leg():
    graph = TravelGraph()
    graph.add_route("A", "B", 50, 5, "auto", toll=10)
    graph.add_route("B", "C", 30, 3, "auto", toll=4)

    assert graph.shortest_path_totals("A")["C"] == PathTotals(cost=80, time=8, legs=2, toll=14)
    assert [leg["toll"] for leg in graph.get_route_details(["A", "B", "C"])] == [10, 4]
