"""Cost and time matrices of a transport mode, with per-segment lookups."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.ui.api_client import get_matrix_from_api

UNREACHABLE = -1.0


@dataclass(frozen=True)
class RouteMatrices:
    """Cost (€) and time (h) matrices sharing the same city order."""

    cities: List[str]
    cost: List[List[float]]
    time: List[List[float]]
    city_index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "city_index", {city: index for index, city in enumerate(self.cities)})

    def missing_cities(self, cities: List[str]) -> List[str]:
        """Return the cities that are not part of the matrices."""
        return [city for city in cities if city not in self.city_index]

    def cost_between(self, origin: str, destination: str) -> Optional[float]:
        """Return the direct cost between two cities, or None when unreachable."""
        return self._lookup(self.cost, origin, destination)

    def time_between(self, origin: str, destination: str) -> Optional[float]:
        """Return the direct time between two cities, or None when unreachable."""
        return self._lookup(self.time, origin, destination)

    def is_connected(self, origin: str, destination: str) -> bool:
        """Return whether both the cost and the time between two cities are known."""
        return self.cost_between(origin, destination) is not None and self.time_between(origin, destination) is not None

    def submatrix(self, cities: List[str], optimize_by: str) -> List[List[float]]:
        """Return the matrix of the optimization criterion restricted to the cities."""
        matrix = self.cost if optimize_by == "cost" else self.time
        indices = [self.city_index[city] for city in cities]
        return [[matrix[row][column] for column in indices] for row in indices]

    def route_cost(self, route: List[str]) -> float:
        """Return the cost in euros of a route, skipping unknown segments."""
        return sum(self.segment_costs(route))

    def segment_costs(self, route: List[str]) -> List[float]:
        """Return the cost of every consecutive segment, 0 when unknown."""
        return [self.cost_between(origin, destination) or 0.0 for origin, destination in zip(route, route[1:])]

    def segment_rows(self, route: List[str]) -> List[Dict]:
        """Return one table row per segment with its cost and time."""
        return [
            {
                "#": number,
                "Origen": origin,
                "Destino": destination,
                "Costo (€)": self.cost_between(origin, destination),
                "Tiempo (h)": self.time_between(origin, destination),
            }
            for number, (origin, destination) in enumerate(zip(route, route[1:]), start=1)
        ]

    def _lookup(self, matrix: List[List[float]], origin: str, destination: str) -> Optional[float]:
        """Return a matrix value between two cities, or None when unknown or unreachable."""
        if origin not in self.city_index or destination not in self.city_index:
            return None
        value = matrix[self.city_index[origin]][self.city_index[destination]]
        return None if value == UNREACHABLE else value


def load_route_matrices(transport_mode: str) -> Optional[RouteMatrices]:
    """Fetch the cost and time matrices of a transport mode from the API."""
    cost_data = get_matrix_from_api(transport_mode, "cost")
    time_data = get_matrix_from_api(transport_mode, "time")
    if not cost_data or not time_data:
        return None
    return RouteMatrices(cities=cost_data["cities"], cost=cost_data["matrix"], time=time_data["matrix"])
