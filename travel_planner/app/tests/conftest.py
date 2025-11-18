"""
conftest.py - Configuración y fixtures para los tests con Pytest.
Define datos de prueba reutilizables para todos los módulos de test.
"""
import pytest
from app.core.graph import TravelGraph
from app.core.itinerary_validator import ItineraryConstraints, TransportType

@pytest.fixture
def simple_graph():
    """
    Crea un grafo simple para pruebas de Dijkstra.
    Estructura: A -> B -> C
    """
    graph = TravelGraph()
    # Ruta A -> B (Costo 10, Tiempo 1)
    graph.add_route("A", "B", cost=10, time=1, transport_type="bus")
    # Ruta B -> C (Costo 20, Tiempo 2)
    graph.add_route("B", "C", cost=20, time=2, transport_type="tren")
    # Ruta A -> C directa (Más cara pero más rápida)
    graph.add_route("A", "C", cost=50, time=0.5, transport_type="avión")
    return graph

@pytest.fixture
def cost_matrix_3x3():
    """
    Matriz de costos para 3 ciudades (Triángulo equilátero asimétrico).
    0: A, 1: B, 2: C
    """
    return [
        [0, 10, 15], # Desde A
        [10, 0, 35], # Desde B
        [15, 35, 0]  # Desde C
    ]

@pytest.fixture
def basic_constraints():
    """Restricciones básicas para validación de itinerarios."""
    return ItineraryConstraints(
        max_budget=1000,
        max_duration_hours=24,
        max_segments=5,
        required_cities=["Madrid", "Barcelona"],
        allowed_transports=[TransportType.AVION, TransportType.TREN]
    )