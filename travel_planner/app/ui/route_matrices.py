"""Cost, time, legs and tolls of the best route between cities of a transport mode, with per-segment lookups."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.ui.api_client import get_matrix_from_api

UNREACHABLE = -1.0


@dataclass(frozen=True)
class RouteMatrices:
    """Cost (€), time (h), legs and tolls (€, part of the cost) of the best route between cities.

    Every matrix shares the same city order. A segment between two cities may chain several legs
    when there is no direct connection.
    """

    cities: List[str]
    cost: List[List[float]]
    time: List[List[float]]
    legs: List[List[float]]
    tolls: List[List[float]]
    city_index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "city_index", {city: index for index, city in enumerate(self.cities)})

    def missing_cities(self, cities: List[str]) -> List[str]:
        """Return the cities that are not part of the matrices."""
        return [city for city in cities if city not in self.city_index]

    def cost_between(self, origin: str, destination: str) -> Optional[float]:
        """Return the cost between two cities, or None when unreachable."""
        return self._lookup(self.cost, origin, destination)

    def time_between(self, origin: str, destination: str) -> Optional[float]:
        """Return the time between two cities, or None when unreachable."""
        return self._lookup(self.time, origin, destination)

    def layovers_between(self, origin: str, destination: str) -> Optional[int]:
        """Return how many connections the route between two cities makes, or None when unreachable."""
        legs = self._lookup(self.legs, origin, destination)
        return None if legs is None else max(int(legs) - 1, 0)

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

    def route_time(self, route: List[str]) -> float:
        """Return the time in hours of a route, skipping unknown segments."""
        return self._sum_along(self.time, route)

    def route_tolls(self, route: List[str]) -> float:
        """Return the tolls in euros included in the cost of a route, skipping unknown segments."""
        return self._sum_along(self.tolls, route)

    def segment_costs(self, route: List[str]) -> List[float]:
        """Return the cost of every consecutive segment, 0 when unknown."""
        return [self.cost_between(origin, destination) or 0.0 for origin, destination in zip(route, route[1:])]

    def segment_rows(self, route: List[str]) -> List[Dict]:
        """Return one table row per segment with its cost, time and connections."""
        return [
            {
                "#": number,
                "Origen": origin,
                "Destino": destination,
                "Costo (€)": self.cost_between(origin, destination),
                "Tiempo (h)": self.time_between(origin, destination),
                "Escalas": self.layovers_between(origin, destination),
            }
            for number, (origin, destination) in enumerate(zip(route, route[1:]), start=1)
        ]

    def _sum_along(self, matrix: List[List[float]], route: List[str]) -> float:
        """Return the matrix values summed over the consecutive segments of a route, skipping unknown ones."""
        return sum(self._lookup(matrix, origin, destination) or 0.0 for origin, destination in zip(route, route[1:]))

    def _lookup(self, matrix: List[List[float]], origin: str, destination: str) -> Optional[float]:
        """Return a matrix value between two cities, or None when unknown or unreachable."""
        if origin not in self.city_index or destination not in self.city_index:
            return None
        value = matrix[self.city_index[origin]][self.city_index[destination]]
        return None if value == UNREACHABLE else value


def load_route_matrices(transport_mode: str, optimize_by: str) -> Optional[RouteMatrices]:
    """Fetch from the API the cost, time, legs and tolls of the best routes of a transport mode for the criterion."""
    data = get_matrix_from_api(transport_mode, optimize_by)
    if not data:
        return None
    return RouteMatrices(
        cities=data["cities"],
        cost=data["cost_matrix"],
        time=data["time_matrix"],
        legs=data["legs_matrix"],
        tolls=data["toll_matrix"],
    )
