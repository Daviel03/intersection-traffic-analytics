from __future__ import annotations

from typing import Iterable

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def bottom_center(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def distance(point_a: Point, point_b: Point) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def polygon_centroid(polygon: Iterable[Point]) -> Point:
    polygon_points = list(polygon)
    if not polygon_points:
        return (0.0, 0.0)
    x = sum(point[0] for point in polygon_points) / len(polygon_points)
    y = sum(point[1] for point in polygon_points) / len(polygon_points)
    return (x, y)


def crosses_line(path_start: Point, path_end: Point, line_start: Point, line_end: Point) -> bool:
    if path_start == path_end:
        return False
    return segments_intersect(path_start, path_end, line_start, line_end)


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    polygon_points = list(polygon)
    if len(polygon_points) < 3:
        return False

    x, y = point
    inside = False
    previous = polygon_points[-1]

    for current in polygon_points:
        if _point_on_segment(previous, point, current):
            return True
        x1, y1 = current
        x2, y2 = previous
        intersects = ((y1 > y) != (y2 > y)) and (
            x < ((x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12)) + x1
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _point_on_segment(a1, b1, a2):
        return True
    if o2 == 0 and _point_on_segment(a1, b2, a2):
        return True
    if o3 == 0 and _point_on_segment(b1, a1, b2):
        return True
    if o4 == 0 and _point_on_segment(b1, a2, b2):
        return True
    return False


def _orientation(point_a: Point, point_b: Point, point_c: Point) -> int:
    value = (
        (point_b[1] - point_a[1]) * (point_c[0] - point_b[0])
        - (point_b[0] - point_a[0]) * (point_c[1] - point_b[1])
    )
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _point_on_segment(segment_start: Point, point: Point, segment_end: Point) -> bool:
    if (
        min(segment_start[0], segment_end[0]) - 1e-9
        <= point[0]
        <= max(segment_start[0], segment_end[0]) + 1e-9
        and min(segment_start[1], segment_end[1]) - 1e-9
        <= point[1]
        <= max(segment_start[1], segment_end[1]) + 1e-9
    ):
        cross = (
            (point[1] - segment_start[1]) * (segment_end[0] - segment_start[0])
            - (point[0] - segment_start[0]) * (segment_end[1] - segment_start[1])
        )
        return abs(cross) < 1e-9
    return False
