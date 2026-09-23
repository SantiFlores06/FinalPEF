"""Text formatting shared by the views."""

from typing import List

ROUTE_SEPARATOR = " → "


def format_route(route: List[str]) -> str:
    """Join the cities of a route with arrows."""
    return ROUTE_SEPARATOR.join(route)


def format_timestamp(iso_timestamp: str) -> str:
    """Turn an ISO timestamp into a readable Spanish date and time."""
    return iso_timestamp.split(".")[0].replace("T", " a las ")
