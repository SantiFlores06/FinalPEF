"""Great-circle arcs of flight legs: the shortest path over the globe instead of a flat line on the map."""

import math
from typing import List, Tuple

Coordinate = Tuple[float, float]
Vector = Tuple[float, float, float]

FULL_TURN_DEGREES = 360.0
HALF_TURN_DEGREES = 180.0
# One interpolated point every few degrees of arc, capped so a long-haul leg stays light to draw
ARC_STEP_DEGREES = 2.0
ARC_MAX_SEGMENTS = 64
# Coincident or antipodal endpoints have no single great circle, so they keep the straight line
DEGENERATE_ARC_SINE = 1e-9


def to_unit_vector(coordinate: Coordinate) -> Vector:
    """Return the point on the unit sphere of a (latitude, longitude) pair."""
    latitude, longitude = (math.radians(value) for value in coordinate)
    return math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)


def to_coordinate(vector: Vector) -> Coordinate:
    """Return the (latitude, longitude) pair of a point on the sphere, with the longitude within ±180°."""
    x, y, z = vector
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def central_angle(start: Vector, end: Vector) -> float:
    """Return the angle in radians between two unit vectors, accurate for near and far points alike."""
    (start_x, start_y, start_z), (end_x, end_y, end_z) = start, end
    cross_norm = math.hypot(
        start_y * end_z - start_z * end_y, start_z * end_x - start_x * end_z, start_x * end_y - start_y * end_x
    )
    return math.atan2(cross_norm, start_x * end_x + start_y * end_y + start_z * end_z)


def slerp(start: Vector, end: Vector, angle: float, fraction: float) -> Vector:
    """Return the point at the given fraction of the arc between two unit vectors (spherical interpolation)."""
    start_weight = math.sin((1 - fraction) * angle) / math.sin(angle)
    end_weight = math.sin(fraction * angle) / math.sin(angle)
    x, y, z = (start_weight * start_axis + end_weight * end_axis for start_axis, end_axis in zip(start, end))
    return x, y, z


def arc_segments(angle: float) -> int:
    """Return how many pieces an arc is split into: more for longer legs, never above ARC_MAX_SEGMENTS."""
    return max(1, min(ARC_MAX_SEGMENTS, math.ceil(math.degrees(angle) / ARC_STEP_DEGREES)))


def unwrap_longitudes(coordinates: List[Coordinate]) -> List[Coordinate]:
    """Shift each longitude by whole turns so consecutive points differ by less than half a turn."""
    unwrapped = [coordinates[0]]
    for latitude, longitude in coordinates[1:]:
        turns = round((unwrapped[-1][1] - longitude) / FULL_TURN_DEGREES)
        unwrapped.append((latitude, longitude + turns * FULL_TURN_DEGREES))
    return unwrapped


def great_circle_path(start: Coordinate, end: Coordinate) -> List[Coordinate]:
    """Return the great-circle arc from start to end as (latitude, longitude) points.

    Longitudes are unwrapped so the arc stays continuous across the antimeridian: a trans-Pacific leg keeps
    growing past 180° (Leaflet draws it fine) instead of jumping back across the whole map.
    """
    start_vector, end_vector = to_unit_vector(start), to_unit_vector(end)
    angle = central_angle(start_vector, end_vector)
    if math.sin(angle) < DEGENERATE_ARC_SINE:
        return [start, end]
    segments = arc_segments(angle)
    inner_points = [
        to_coordinate(slerp(start_vector, end_vector, angle, step / segments)) for step in range(1, segments)
    ]
    return unwrap_longitudes([start, *inner_points, end])


def shift_longitudes(path: List[Coordinate], degrees: float) -> List[Coordinate]:
    """Return the path moved east (positive) or west (negative) by the given degrees of longitude."""
    return [(latitude, longitude + degrees) for latitude, longitude in path]


def world_copies(path: List[Coordinate]) -> List[List[Coordinate]]:
    """Return the path plus, when it crosses the antimeridian, its copy one full turn back.

    The markers stay at their real longitudes, so an arc drawn from its origin across the antimeridian ends on
    another copy of the world; the shifted copy is the same arc arriving at the destination's real marker.
    """
    longitudes = [longitude for _, longitude in path]
    if max(longitudes) > HALF_TURN_DEGREES:
        return [path, shift_longitudes(path, -FULL_TURN_DEGREES)]
    if min(longitudes) < -HALF_TURN_DEGREES:
        return [path, shift_longitudes(path, FULL_TURN_DEGREES)]
    return [path]
