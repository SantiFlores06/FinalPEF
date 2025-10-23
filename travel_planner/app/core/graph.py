# travel_planner/app/core/graph.py
"""
graph.py - Implementación de Dijkstra desde cero.
Algoritmo de camino mínimo para el sistema de planificación de viajes.
"""

import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Route:
    """Representa una ruta entre dos ciudades."""
    destination: str
    cost: float
    time: float
    transport_type: str


class TravelGraph:
    """
    Grafo dirigido ponderado para representar ciudades y rutas de transporte.
    
    Implementa el algoritmo de Dijkstra desde cero para encontrar 
    caminos mínimos entre ciudades.
    """
    
    def __init__(self) -> None:
        """Inicializa un grafo vacío usando lista de adyacencia."""
        self.graph: Dict[str, List[Route]] = {}
        self.vertices: set = set()
    
    def add_vertex(self, vertex: str) -> None:
        """
        Agrega un vértice (ciudad) al grafo.
        
        Args:
            vertex: Nombre de la ciudad.
        """
        if vertex not in self.graph:
            self.graph[vertex] = []
            self.vertices.add(vertex)
    
    def add_route(
        self, 
        origin: str, 
        destination: str, 
        cost: float, 
        time: float, 
        transport_type: str
    ) -> None:
        """
        Agrega una ruta al grafo con múltiples atributos.
        
        Args:
            origin: Ciudad de origen.
            destination: Ciudad de destino.
            cost: Costo de la ruta en unidades monetarias.
            time: Tiempo de viaje en horas.
            transport_type: Tipo de transporte (bus, tren, avión, etc.).
        """
        self.add_vertex(origin)
        self.add_vertex(destination)
        
        route = Route(
            destination=destination,
            cost=cost,
            time=time,
            transport_type=transport_type
        )
        self.graph[origin].append(route)
    
    def dijkstra(
        self, 
        source: str, 
        target: Optional[str] = None,
        weight: str = 'cost'
    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Implementación del algoritmo de Dijkstra desde cero.
        
        Encuentra el camino más corto desde el nodo origen a todos los demás
        nodos (o a un nodo específico si se proporciona target).
        
        Args:
            source: Nodo de origen.
            target: Nodo de destino opcional. Si es None, calcula a todos.
            weight: Atributo por el cual ponderar ('cost' o 'time').
        
        Returns:
            Tupla con dos diccionarios:
            - distances: Distancias mínimas desde source a cada nodo.
            - predecessors: Predecesor de cada nodo en el camino más corto.
        
        Complejidad:
            Tiempo: O((V + E) log V) con cola de prioridad.
            Espacio: O(V) para estructuras auxiliares.
        """
        if source not in self.vertices:
            raise ValueError(f"Nodo origen '{source}' no existe en el grafo")
        
        # Inicialización
        distances: Dict[str, float] = {vertex: float('inf') for vertex in self.vertices}
        predecessors: Dict[str, Optional[str]] = {vertex: None for vertex in self.vertices}
        distances[source] = 0
        
        # Cola de prioridad: (distancia, nodo)
        priority_queue: List[Tuple[float, str]] = [(0, source)]
        
        # Conjunto de nodos visitados/finalizados
        visited: set = set()
        
        while priority_queue:
            # Extraer nodo con menor distancia
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # Si ya fue procesado, saltar (entrada obsoleta en la cola)
            if current_node in visited:
                continue
            
            # Marcar como visitado
            visited.add(current_node)
            
            # Optimización: si llegamos al target, podemos terminar
            if target and current_node == target:
                break
            
            # Saltar si la distancia extraída está desactualizada
            if current_distance > distances[current_node]:
                continue
            
            # Relajación de aristas
            for route in self.graph[current_node]:
                neighbor = route.destination
                
                # Obtener el peso según el criterio especificado
                edge_weight = getattr(route, weight)
                
                # Calcular nueva distancia tentativa
                new_distance = current_distance + edge_weight
                
                # Si encontramos un camino más corto, actualizamos
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        return distances, predecessors
    
    def find_shortest_path(
        self, 
        origin: str, 
        destination: str,
        weight: str = 'cost'
    ) -> Tuple[List[str], float]:
        """
        Encuentra el camino más corto entre dos nodos.
        
        Args:
            origin: Ciudad de origen.
            destination: Ciudad de destino.
            weight: Criterio de optimización ('cost' o 'time').
        
        Returns:
            Tupla con:
            - path: Lista de ciudades en el camino óptimo.
            - distance: Costo/tiempo total del camino.
        """
        if destination not in self.vertices:
            return [], float('inf')
        
        # Ejecutar Dijkstra
        distances, predecessors = self.dijkstra(origin, destination, weight)
        
        # Reconstruir el camino desde destination hacia origin
        path = self._reconstruct_path(predecessors, origin, destination)
        
        return path, distances[destination]
    
    def _reconstruct_path(
        self,
        predecessors: Dict[str, Optional[str]],
        origin: str,
        destination: str
    ) -> List[str]:
        """
        Reconstruye el camino desde origin hasta destination.
        
        Args:
            predecessors: Diccionario de predecesores de Dijkstra.
            origin: Nodo de inicio.
            destination: Nodo de destino.
        
        Returns:
            Lista de nodos que forman el camino desde origin a destination.
            Lista vacía si no hay camino.
        """
        path = []
        current = destination
        
        # Recorrer hacia atrás desde destination hasta origin
        while current is not None:
            path.append(current)
            current = predecessors[current]
            
            # Prevenir bucles infinitos
            if len(path) > len(self.vertices):
                return []
        
        # Invertir para obtener el camino de origin a destination
        path.reverse()
        
        # Verificar que el camino comienza en origin
        if path[0] != origin:
            return []
        
        return path
    
    def find_all_shortest_paths(
        self, 
        origin: str,
        weight: str = 'cost'
    ) -> Dict[str, Tuple[List[str], float]]:
        """
        Encuentra todos los caminos más cortos desde un origen a todos los destinos.
        
        Args:
            origin: Ciudad de origen.
            weight: Criterio de optimización.
        
        Returns:
            Diccionario donde la clave es el destino y el valor es una tupla
            con (camino, distancia).
        """
        distances, predecessors = self.dijkstra(origin, weight=weight)
        
        results = {}
        for destination in self.vertices:
            if destination == origin:
                continue
            
            path = self._reconstruct_path(predecessors, origin, destination)
            results[destination] = (path, distances[destination])
        
        return results
    
    def get_route_details(self, path: List[str]) -> List[Dict]:
        """
        Obtiene los detalles completos de las rutas en un camino.
        
        Args:
            path: Lista de ciudades en orden.
        
        Returns:
            Lista de diccionarios con información de cada segmento.
        """
        details = []
        
        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]
            
            # Buscar la ruta en el grafo
            for route in self.graph[origin]:
                if route.destination == destination:
                    details.append({
                        'from': origin,
                        'to': destination,
                        'cost': route.cost,
                        'time': route.time,
                        'transport': route.transport_type
                    })
                    break
        
        return details


# Ejemplo de uso
if __name__ == "__main__":
    # Crear grafo de ejemplo
    graph = TravelGraph()
    
    # Agregar rutas
    graph.add_route("Madrid", "Barcelona", 50, 3, "tren")
    graph.add_route("Madrid", "Valencia", 40, 4, "bus")
    graph.add_route("Barcelona", "París", 100, 2, "avión")
    graph.add_route("Valencia", "Barcelona", 45, 3.5, "tren")
    graph.add_route("Valencia", "París", 120, 8, "bus")
    graph.add_route("Madrid", "París", 150, 2.5, "avión")
    
    # Encontrar camino más corto por costo
    path_cost, cost = graph.find_shortest_path("Madrid", "París", weight='cost')
    print(f"Camino óptimo por costo: {' -> '.join(path_cost)}")
    print(f"Costo total: {cost}€\n")
    
    # Encontrar camino más corto por tiempo
    path_time, time = graph.find_shortest_path("Madrid", "París", weight='time')
    print(f"Camino óptimo por tiempo: {' -> '.join(path_time)}")
    print(f"Tiempo total: {time} horas\n")
    
    # Detalles de la ruta
    details = graph.get_route_details(path_cost)
    print("Detalles del viaje:")
    for segment in details:
        print(f"  {segment['from']} → {segment['to']}: "
              f"{segment['cost']}€, {segment['time']}h, {segment['transport']}")
