"""
test_tsp.py - Tests para el algoritmo TSP (Travelling Salesman Problem).
"""
import pytest
from app.core.tsp_dp import TSPSolver

def test_tsp_3_cities(cost_matrix_3x3):
    """
    Prueba TSP con 3 ciudades (Tour cerrado).
    Ruta esperada: 0 -> 1 -> 2 -> 0
    Costo: 10 (0->1) + 35 (1->2) + 15 (2->0) = 60
    """
    solver = TSPSolver(cost_matrix_3x3)
    min_cost, route = solver.solve(start_city=0, return_to_start=True)
    
    assert len(route) == 4 # 3 ciudades + regreso al inicio
    assert route[0] == route[-1] == 0 # Empieza y termina en 0
    assert min_cost == 60

def test_tsp_open_tour(cost_matrix_3x3):
    """
    Prueba TSP sin regresar al inicio (Hamiltonian Path).
    Mejor ruta: 0 -> 1 -> 2
    Costo: 10 + 35 = 45
    (Nota: 0 -> 2 -> 1 sería 15 + 35 = 50)
    """
    solver = TSPSolver(cost_matrix_3x3)
    min_cost, route = solver.solve(start_city=0, return_to_start=False)
    
    assert len(route) == 3 # Solo visita las 3 ciudades
    assert route[-1] != 0 # No regresa
    assert min_cost == 45

def test_invalid_matrix():
    """Debe lanzar error si la matriz no es cuadrada."""
    invalid_matrix = [[0, 1], [0, 1], [0, 1]] # 3x2
    with pytest.raises(ValueError):
        TSPSolver(invalid_matrix)

def test_get_route_names():
    """Verifica la conversión de índices a nombres."""
    matrix = [[0, 10], [10, 0]]
    names = ["Casa", "Trabajo"]
    solver = TSPSolver(matrix, city_names=names)
    
    # Ruta 0 -> 1
    path_names = solver.get_route_with_names([0, 1])
    assert path_names == ["Casa", "Trabajo"]