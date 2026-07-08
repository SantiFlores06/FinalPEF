"""
tsp_genetic.py - TSP solver con Algoritmos Genéticos.
Heurístico que escala a 20+ ciudades donde Held-Karp se vuelve inviable.

Complejidad: O(generaciones × población × n)
Calidad: ~5-15% del óptimo dependiendo de parámetros.
"""

import random
import time
from typing import Dict, List, Optional, Tuple


class GeneticTSP:
    """
    Resuelve el TSP con Algoritmos Genéticos.

    Codificación:
        Cromosoma: permutación de índices de ciudades.
        Ej: [0, 3, 1, 2] = Madrid -> Viena -> Barcelona -> Roma -> (Madrid)

    Operadores:
        Selección: Torneo (tournament selection)
        Cruce: Order Crossover (OX) — preserva permutaciones válidas
        Mutación: Swap de dos ciudades aleatorias

    Fitness: 1 / costo_total (minimizar costo = maximizar fitness)

    A diferencia de Held-Karp (O(n² × 2^n)), este escala a 100+ ciudades.
    No garantiza el óptimo, pero converge a soluciones muy buenas.
    """

    def __init__(
        self,
        cost_matrix: List[List[float]],
        city_names: Optional[List[str]] = None,
        population_size: int = 150,
        generations: int = 300,
        mutation_rate: float = 0.02,
        tournament_size: int = 5,
        elitism: int = 2,
    ) -> None:
        """
        Args:
            cost_matrix: Matriz n×n de costos entre ciudades.
            city_names: Nombres de ciudades (opcional).
            population_size: Individuos por generación.
            generations: Número de generaciones a evolucionar.
            mutation_rate: Probabilidad de mutación por individuo (0-1).
            tournament_size: Candidatos por torneo de selección.
            elitism: Cuántos mejores individuos pasan directo a la siguiente generación.
        """
        if not cost_matrix or len(cost_matrix) != len(cost_matrix[0]):
            raise ValueError("La matriz de costos debe ser cuadrada y no vacía")

        self.n = len(cost_matrix)
        self.cost = cost_matrix
        self.city_names = city_names if city_names else [f"Ciudad_{i}" for i in range(self.n)]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = min(tournament_size, population_size)
        self.elitism = elitism

        # Historial por generación — para graficar convergencia
        self.history: List[Dict] = []
        self.elapsed_ms: float = 0.0

    # ------------------------------------------------------------------
    # OPERADORES INTERNOS
    # ------------------------------------------------------------------

    def _route_cost(self, route: List[int], return_to_start: bool) -> float:
        """Calcula el costo total de un cromosoma."""
        total = 0.0
        for i in range(len(route) - 1):
            c = self.cost[route[i]][route[i + 1]]
            if c < 0 or c == float("inf"):
                return float("inf")
            total += c
        if return_to_start and self.n > 1:
            c = self.cost[route[-1]][route[0]]
            if c < 0 or c == float("inf"):
                return float("inf")
            total += c
        return total

    def _create_individual(self, start_city: int) -> List[int]:
        """Crea un cromosoma aleatorio válido (permutación)."""
        others = [i for i in range(self.n) if i != start_city]
        random.shuffle(others)
        return [start_city] + others

    def _tournament_select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Selección por torneo: elige el mejor de k candidatos aleatorios."""
        k = min(self.tournament_size, len(population))
        candidates = random.sample(range(len(population)), k)
        winner = min(candidates, key=lambda i: costs[i])
        return population[winner][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX): produce descendientes con permutaciones válidas.

        Toma un segmento de parent1 y rellena el resto con el orden de parent2.
        El índice 0 (ciudad inicial) siempre se mantiene fijo.

        Ej:
            parent1: [0, 1, 2, 3, 4]
            parent2: [0, 3, 1, 4, 2]
            segmento [1:3] → [1, 2]
            hijo:    [0, _, 1, 2, _] → relleno de parent2 → [0, 3, 1, 2, 4]
        """
        n = len(parent1)
        if n <= 2:
            return parent1[:]

        a, b = sorted(random.sample(range(1, n), 2))

        child = [-1] * n
        child[0] = parent1[0]
        child[a:b] = parent1[a:b]

        segment_set = set(child[a:b]) | {parent1[0]}
        fill = [x for x in parent2 if x not in segment_set]

        pos = 0
        for i in range(1, n):
            if child[i] == -1:
                child[i] = fill[pos]
                pos += 1

        return child

    def _swap_mutate(self, individual: List[int]) -> List[int]:
        """Mutación swap: intercambia dos ciudades aleatorias (sin tocar la ciudad inicial)."""
        if len(individual) <= 2:
            return individual
        mutant = individual[:]
        i, j = random.sample(range(1, len(mutant)), 2)
        mutant[i], mutant[j] = mutant[j], mutant[i]
        return mutant

    # ------------------------------------------------------------------
    # ALGORITMO PRINCIPAL
    # ------------------------------------------------------------------

    def solve(
        self,
        start_city: int = 0,
        return_to_start: bool = True,
        progress_callback=None,
    ) -> Tuple[float, List[int]]:
        """
        Ejecuta el algoritmo genético y retorna la mejor solución encontrada.

        Args:
            start_city: Índice de la ciudad de partida.
            return_to_start: Si True, el tour regresa a la ciudad inicial.

        Returns:
            (mejor_costo, mejor_ruta) — misma interfaz que TSPSolver.solve().
        """
        self.history = []
        t0 = time.perf_counter()

        # Población inicial aleatoria
        population = [self._create_individual(start_city) for _ in range(self.population_size)]
        costs = [self._route_cost(ind, return_to_start) for ind in population]

        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        best_route = population[best_idx][:]
        best_cost = costs[best_idx]

        for gen in range(self.generations):
            # Ordenar por costo (mejor = menor costo)
            pairs = sorted(zip(costs, population), key=lambda x: x[0])
            population = [ind for _, ind in pairs]
            costs = [c for c, _ in pairs]

            if costs[0] < best_cost:
                best_cost = costs[0]
                best_route = population[0][:]

            self.history.append({
                "generation": gen + 1,
                "best_cost": round(best_cost, 2),
                "avg_cost": round(sum(costs) / len(costs), 2),
            })
            if progress_callback:
                progress_callback(gen + 1, self.generations, best_cost)

            # Construir siguiente generación
            next_pop: List[List[int]] = population[: self.elitism]  # elitismo

            while len(next_pop) < self.population_size:
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)
                child = self._order_crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._swap_mutate(child)
                next_pop.append(child)

            population = next_pop
            costs = [self._route_cost(ind, return_to_start) for ind in population]

        self.elapsed_ms = (time.perf_counter() - t0) * 1000

        route = best_route[:]
        if return_to_start:
            route.append(start_city)

        return best_cost, route

    def get_route_with_names(self, route: List[int]) -> List[str]:
        """Convierte índices de ciudades a nombres."""
        return [self.city_names[i] for i in route]
