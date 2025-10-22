# travel_planner/app/core/graph.py
"""
    Aca esta la implementacion del grafo y el algoritmo de Dijkstra
    para encontrar la ruta mas corta entre dos puntos.
"""
import heapq

class Graph:
    def __init__(self):
        # Diccionario de adyacencia: {origen: [(destino, costo, tiempo), ...]}
        self.routes = {}

    def add_route(self, origin, destination, cost, time):
        """Agrega una conexión entre dos puntos."""
        if origin not in self.routes:
            self.routes[origin] = []
        self.routes[origin].append((destination, cost, time))

    def get_neighbors(self, node):
        """Devuelve las rutas disponibles desde un nodo."""
        return self.routes.get(node, [])

    def dijkstra(self, start, end, weight="cost"):
        """
        Calcula el camino mínimo entre start y end usando Dijkstra.
        weight puede ser 'cost' o 'time'.
        """
        if weight not in ("cost", "time"):
            raise ValueError("weight debe ser 'cost' o 'time'")

        # índice: 1 = costo, 2 = tiempo
        idx = 1 if weight == "cost" else 2

        queue = [(0, start, [])]  # (distancia acumulada, nodo actual, camino)
        visited = set()

        while queue:
            (accum, node, path) = heapq.heappop(queue)

            if node in visited:
                continue
            visited.add(node)

            path = path + [node]

            if node == end:
                return {"total": accum, "path": path}

            for (neighbor, cost, time) in self.get_neighbors(node):
                value = cost if weight == "cost" else time
                heapq.heappush(queue, (accum + value, neighbor, path))

        return None  # No hay ruta



# Prueba rápida si ejecutas directamente este archivo
if __name__ == "__main__":
    g = Graph()
    g.add_route("Buenos Aires", "Córdoba", cost=100, time=2)
    g.add_route("Córdoba", "Mendoza", cost=120, time=3)
    g.add_route("Buenos Aires", "Rosario", cost=50, time=1)
    g.add_route("Rosario", "Mendoza", cost=180, time=4)

    result = g.dijkstra("Buenos Aires", "Mendoza", weight="cost")
    print(result)
