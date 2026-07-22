"""
test_tsp_genetic.py - Tests para el TSP con Algoritmo Genético (GeneticTSP).

El algoritmo es estocástico: usa el módulo `random` global. Fijando la semilla
(`random.seed(N)`) antes de cada `solve()` se vuelve determinista, lo que permite
testear tanto reproducibilidad como calidad de la solución.
"""
import math
import random

import pytest

from app.core.tsp_genetic import GeneticTSP
from app.core.tsp_dp import TSPSolver


@pytest.fixture
def cost_matrix_5():
    """
    Matriz simétrica de 5 ciudades para los tests del genético.
    El óptimo es único y pequeño, así que el genético debería alcanzarlo.
    """
    return [
        [0, 12, 10, 19, 8],
        [12, 0, 3, 7, 2],
        [10, 3, 0, 6, 20],
        [19, 7, 6, 0, 4],
        [8, 2, 20, 4, 0],
    ]


def test_tour_is_valid_permutation(cost_matrix_5):
    """El tour visita todas las ciudades exactamente una vez y cierra en el origen."""
    random.seed(1)
    ga = GeneticTSP(cost_matrix_5, population_size=50, generations=50)
    _, route = ga.solve(start_city=0, return_to_start=True)

    # Con return_to_start el tour termina volviendo al inicio.
    assert route[0] == route[-1] == 0
    visited = route[:-1]  # sin el regreso al origen
    assert sorted(visited) == [0, 1, 2, 3, 4]  # permutación completa, sin repetidos


def test_determinism_with_fixed_seed(cost_matrix_5):
    """Con la misma semilla, dos ejecuciones producen el mismo costo y la misma ruta."""
    ga = GeneticTSP(cost_matrix_5, population_size=60, generations=80)

    random.seed(123)
    cost1, route1 = ga.solve(start_city=0, return_to_start=True)

    random.seed(123)
    cost2, route2 = ga.solve(start_city=0, return_to_start=True)

    assert cost1 == cost2
    assert route1 == route2


def test_reaches_held_karp_optimum(cost_matrix_5):
    """En 5 ciudades el genético alcanza el óptimo exacto calculado por Held-Karp."""
    optimal_cost, _ = TSPSolver(cost_matrix_5).solve(start_city=0, return_to_start=True)

    random.seed(7)
    ga = GeneticTSP(cost_matrix_5, population_size=120, generations=200)
    ga_cost, _ = ga.solve(start_city=0, return_to_start=True)

    # El genético nunca puede mejorar al óptimo; con estos parámetros debe igualarlo.
    assert ga_cost == pytest.approx(optimal_cost)


def test_history_best_cost_is_monotonic(cost_matrix_5):
    """El mejor costo por generación nunca empeora (verifica el elitismo)."""
    random.seed(5)
    ga = GeneticTSP(cost_matrix_5, population_size=50, generations=60)
    ga.solve(start_city=0, return_to_start=True)

    assert len(ga.history) == 60
    best_costs = [entry["best_cost"] for entry in ga.history]
    for prev, curr in zip(best_costs, best_costs[1:]):
        assert curr <= prev  # monótono no creciente


def test_single_city():
    """Caso borde: una sola ciudad. El costo del tour es 0 y no debe romper."""
    random.seed(0)
    ga = GeneticTSP([[0]], population_size=10, generations=5)
    cost, route = ga.solve(start_city=0, return_to_start=True)

    assert cost == 0
    assert route[0] == 0


def test_two_cities():
    """Caso borde: dos ciudades. Tour cerrado 0->1->0 con costo ida y vuelta."""
    random.seed(0)
    ga = GeneticTSP([[0, 5], [5, 0]], population_size=10, generations=10)
    cost, route = ga.solve(start_city=0, return_to_start=True)

    assert cost == 10  # 5 (0->1) + 5 (1->0)
    assert sorted(route[:-1]) == [0, 1]


def test_unreachable_city():
    """Caso borde: una ciudad inalcanzable (inf) hace que todo tour tenga costo infinito."""
    inf = float("inf")
    matrix = [
        [0, 5, inf],
        [5, 0, inf],
        [inf, inf, 0],
    ]
    random.seed(0)
    ga = GeneticTSP(matrix, population_size=20, generations=20)
    cost, _ = ga.solve(start_city=0, return_to_start=True)

    assert math.isinf(cost)
