"""
tsp_dp.py - Travelling Salesman Problem con Programación Dinámica.
Implementa el algoritmo de Held-Karp usando bitmasking y memoización.
"""

from typing import Dict, List, Tuple, Optional
import sys


class TSPSolver:
    """
    Resuelve el Problema del Viajante (TSP) usando Programación Dinámica.
    
    Implementa el algoritmo de Held-Karp con bitmasking para representar
    subconjuntos de ciudades visitadas. Usa memoización para evitar 
    recalcular subproblemas.
    
    Complejidad:
        Tiempo: O(n² × 2^n)
        Espacio: O(n × 2^n)
    
    Adecuado para n ≤ 20 ciudades.
    """
    
    def __init__(
        self, 
        cost_matrix: List[List[float]], 
        city_names: Optional[List[str]] = None
    ) -> None:
        """
        Inicializa el solver con una matriz de costos.
        
        Args:
            cost_matrix: Matriz nxn donde cost_matrix[i][j] es el costo
                        de ir de la ciudad i a la ciudad j.
            city_names: Nombres opcionales de las ciudades para mejor legibilidad.
        
        Raises:
            ValueError: Si la matriz no es cuadrada o está vacía.
        """
        if not cost_matrix or len(cost_matrix) != len(cost_matrix[0]):
            raise ValueError("La matriz de costos debe ser cuadrada y no vacía")
        
        self.n = len(cost_matrix)
        self.cost = cost_matrix
        self.city_names = city_names if city_names else [f"Ciudad_{i}" for i in range(self.n)]
        
        # Memoización: (mask, pos) -> costo mínimo
        self.memo: Dict[Tuple[int, int], float] = {}
        
        # Para reconstruir el camino
        self.parent: Dict[Tuple[int, int], int] = {}
    
    def solve(
        self, 
        start_city: int = 0, 
        return_to_start: bool = True
    ) -> Tuple[float, List[int]]:
        """
        Encuentra el tour óptimo del TSP.
        
        Args:
            start_city: Índice de la ciudad inicial (0 por defecto).
            return_to_start: Si True, el tour regresa a la ciudad inicial.
        
        Returns:
            Tupla (costo_mínimo, ruta_óptima):
                - costo_mínimo: Costo total del tour óptimo.
                - ruta_óptima: Lista de índices de ciudades en orden de visita.
        """
        self.start_city = start_city
        self.return_to_start = return_to_start
        self.memo.clear()
        self.parent.clear()
        
        # Mask inicial: solo la ciudad de inicio está visitada
        initial_mask = 1 << start_city
        
        # Calcular costo mínimo
        min_cost = self._tsp(initial_mask, start_city)
        
        # Reconstruir la ruta
        route = self._reconstruct_path()
        
        return min_cost, route
    
    def _tsp(self, mask: int, pos: int) -> float:
        """
        Función recursiva con memoización para calcular el costo mínimo.
        
        Args:
            mask: Bitmask que representa el conjunto de ciudades visitadas.
                  Si el bit i está en 1, la ciudad i ha sido visitada.
            pos: Posición/ciudad actual.
        
        Returns:
            Costo mínimo para completar el tour desde el estado (mask, pos).
        
        Ejemplo de bitmasking:
            mask = 13 = 0b1101
            Ciudades visitadas: 0, 2, 3
            Ciudades no visitadas: 1
        """
        # Caso base: todas las ciudades visitadas
        if mask == (1 << self.n) - 1:
            # Si debemos regresar al inicio, agregar ese costo
            if self.return_to_start:
                return self.cost[pos][self.start_city]
            else:
                return 0
        
        # Consultar memoización
        if (mask, pos) in self.memo:
            return self.memo[(mask, pos)]
        
        min_cost = float('inf')
        best_next_city = -1
        
        # Probar visitar cada ciudad no visitada
        for next_city in range(self.n):
            # Verificar si la ciudad ya fue visitada (bit está en 1)
            if mask & (1 << next_city):
                continue
            
            # Costo de ir a next_city + costo del subproblema restante
            new_mask = mask | (1 << next_city)
            cost = self.cost[pos][next_city] + self._tsp(new_mask, next_city)
            
            # Actualizar si encontramos mejor costo
            if cost < min_cost:
                min_cost = cost
                best_next_city = next_city
        
        # Guardar en memo y parent para reconstrucción
        self.memo[(mask, pos)] = min_cost
        if best_next_city != -1:
            self.parent[(mask, pos)] = best_next_city
        
        return min_cost
    
    def _reconstruct_path(self) -> List[int]:
        """
        Reconstruye la ruta óptima usando la información en self.parent.
        
        Returns:
            Lista de índices de ciudades en el orden del tour óptimo.
        """
        route = [self.start_city]
        mask = 1 << self.start_city
        current = self.start_city
        
        # Reconstruir el camino siguiendo las decisiones óptimas
        while mask != (1 << self.n) - 1:
            if (mask, current) not in self.parent:
                break
            
            next_city = self.parent[(mask, current)]
            route.append(next_city)
            mask |= (1 << next_city)
            current = next_city
        
        # Si regresamos al inicio, agregarlo
        if self.return_to_start:
            route.append(self.start_city)
        
        return route
    
    def get_route_with_names(self, route: List[int]) -> List[str]:
        """
        Convierte una ruta de índices a nombres de ciudades.
        
        Args:
            route: Lista de índices de ciudades.
        
        Returns:
            Lista de nombres de ciudades.
        """
        return [self.city_names[i] for i in route]
    
    def get_tour_details(self, route: List[int]) -> List[Dict]:
        """
        Obtiene detalles completos del tour.
        
        Args:
            route: Ruta como lista de índices.
        
        Returns:
            Lista de diccionarios con información de cada segmento.
        """
        details = []
        total_cost = 0
        
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            segment_cost = self.cost[from_city][to_city]
            total_cost += segment_cost
            
            details.append({
                'from': self.city_names[from_city],
                'to': self.city_names[to_city],
                'cost': segment_cost,
                'cumulative_cost': total_cost
            })
        
        return details


class TSPMultiObjective(TSPSolver):
    """
    Extensión de TSPSolver para optimización multi-objetivo.
    
    Permite optimizar por múltiples criterios (costo, tiempo, etc.)
    usando una función de utilidad combinada.
    """
    
    def __init__(
        self, 
        cost_matrix: List[List[float]], 
        time_matrix: List[List[float]],
        city_names: Optional[List[str]] = None,
        cost_weight: float = 0.5,
        time_weight: float = 0.5
    ) -> None:
        """
        Inicializa con múltiples matrices de objetivos.
        
        Args:
            cost_matrix: Matriz de costos.
            time_matrix: Matriz de tiempos.
            city_names: Nombres de ciudades.
            cost_weight: Peso para el costo (0-1).
            time_weight: Peso para el tiempo (0-1).
        """
        # Crear matriz combinada ponderada
        n = len(cost_matrix)
        combined_matrix = [
            [
                cost_weight * cost_matrix[i][j] + time_weight * time_matrix[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        
        super().__init__(combined_matrix, city_names)
        self.cost_matrix_original = cost_matrix
        self.time_matrix_original = time_matrix
    
    def get_tour_metrics(self, route: List[int]) -> Dict[str, float]:
        """
        Calcula métricas del tour en ambas dimensiones.
        
        Args:
            route: Ruta del tour.
        
        Returns:
            Diccionario con costo total y tiempo total.
        """
        total_cost = 0.0
        total_time = 0.0
        
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            total_cost += self.cost_matrix_original[from_city][to_city]
            total_time += self.time_matrix_original[from_city][to_city]
        
        return {
            'total_cost': total_cost,
            'total_time': total_time,
            'num_cities': len(set(route))
        }


class BitMaskUtils:
    """Utilidades para trabajar con bitmasks en TSP."""
    
    @staticmethod
    def is_visited(mask: int, city: int) -> bool:
        """Verifica si una ciudad está en el conjunto visitado."""
        return (mask & (1 << city)) != 0
    
    @staticmethod
    def visit_city(mask: int, city: int) -> int:
        """Marca una ciudad como visitada."""
        return mask | (1 << city)
    
    @staticmethod
    def unvisit_city(mask: int, city: int) -> int:
        """Marca una ciudad como no visitada."""
        return mask & ~(1 << city)
    
    @staticmethod
    def count_visited(mask: int) -> int:
        """Cuenta cuántas ciudades están visitadas."""
        return bin(mask).count('1')
    
    @staticmethod
    def get_visited_cities(mask: int, n: int) -> List[int]:
        """Retorna lista de índices de ciudades visitadas."""
        return [i for i in range(n) if BitMaskUtils.is_visited(mask, i)]
    
    @staticmethod
    def get_unvisited_cities(mask: int, n: int) -> List[int]:
        """Retorna lista de índices de ciudades no visitadas."""
        return [i for i in range(n) if not BitMaskUtils.is_visited(mask, i)]


# Ejemplo de uso
if __name__ == "__main__":
    # Matriz de costos de ejemplo (simétrica)
    cost_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    city_names = ["Madrid", "Barcelona", "Valencia", "Sevilla"]
    
    # Resolver TSP clásico (tour cerrado)
    print("=" * 60)
    print("TSP con Tour Cerrado (regresa al inicio)")
    print("=" * 60)
    
    solver = TSPSolver(cost_matrix, city_names)
    min_cost, route = solver.solve(start_city=0, return_to_start=True)
    
    print(f"\nCosto mínimo: {min_cost}")
    print(f"Ruta (índices): {route}")
    print(f"Ruta (nombres): {' → '.join(solver.get_route_with_names(route))}")
    
    print("\nDetalles del tour:")
    for segment in solver.get_tour_details(route):
        print(f"  {segment['from']} → {segment['to']}: "
              f"costo={segment['cost']}, acumulado={segment['cumulative_cost']}")
    
    # Resolver TSP sin regresar (Hamiltonian Path)
    print("\n" + "=" * 60)
    print("TSP sin Regresar al Inicio (Hamiltonian Path)")
    print("=" * 60)
    
    min_cost2, route2 = solver.solve(start_city=0, return_to_start=False)
    print(f"\nCosto mínimo: {min_cost2}")
    print(f"Ruta: {' → '.join(solver.get_route_with_names(route2))}")
    
    # Ejemplo multi-objetivo
    print("\n" + "=" * 60)
    print("TSP Multi-Objetivo (Costo + Tiempo)")
    print("=" * 60)
    
    time_matrix = [
        [0, 2, 3, 4],
        [2, 0, 5, 4],
        [3, 5, 0, 3],
        [4, 4, 3, 0]
    ]
    
    multi_solver = TSPMultiObjective(
        cost_matrix, 
        time_matrix, 
        city_names,
        cost_weight=0.6,
        time_weight=0.4
    )
    
    min_combined, route3 = multi_solver.solve(start_city=0)
    metrics = multi_solver.get_tour_metrics(route3)
    
    print(f"\nRuta óptima: {' → '.join(multi_solver.get_route_with_names(route3))}")
    print(f"Costo total: {metrics['total_cost']}")
    print(f"Tiempo total: {metrics['total_time']} horas")
    
    # Demostración de operaciones con bitmask
    print("\n" + "=" * 60)
    print("Operaciones con Bitmask")
    print("=" * 60)
    
    mask = 0b1101  # Ciudades 0, 2, 3 visitadas
    print(f"\nMask: {mask} (binario: {bin(mask)})")
    print(f"Ciudades visitadas: {BitMaskUtils.get_visited_cities(mask, 4)}")
    print(f"Ciudades no visitadas: {BitMaskUtils.get_unvisited_cities(mask, 4)}")
    print(f"Total visitadas: {BitMaskUtils.count_visited(mask)}")
