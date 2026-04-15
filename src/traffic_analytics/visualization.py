from __future__ import annotations

from collections import deque

import numpy as np

from traffic_analytics.analytics import AnalyticsEngine
from traffic_analytics.config import RuntimeConfig
from traffic_analytics.geometry import polygon_centroid
from traffic_analytics.io_utils import _require_cv2
from traffic_analytics.tracker_backend import TrackedObject


def render_scene_preview(
    frame,
    config: RuntimeConfig,
    frame_idx: int | None = None,
) -> np.ndarray:
    cv2 = _require_cv2()
    annotated = frame.copy()
    zone_colors = _build_zone_color_map(config)
    _draw_scene_geometry(annotated, config, zone_colors)

    overlay_lines = [f"Scene: {config.output_name}"]
    if frame_idx is not None:
        overlay_lines.append(f"Frame: {frame_idx}")

    y = 25
    for text in overlay_lines:
        cv2.putText(
            annotated,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return annotated


def render_annotated_frame(
    frame,
    tracks: list[TrackedObject],
    analytics: AnalyticsEngine,
    config: RuntimeConfig,
    frame_idx: int,
    track_histories: dict[int, deque[tuple[float, float]]],
    save_trails: bool,
) -> np.ndarray:
    annotated = frame.copy()
    zone_colors = _build_zone_color_map(config)
    cv2 = _require_cv2()
    _draw_scene_geometry(annotated, config, zone_colors)

    for track in tracks:
        entry_zone = analytics.get_track_entry_zone(track.track_id)
        color = zone_colors.get(entry_zone, _track_color(track.track_id))
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        point_x, point_y = (int(value) for value in track.point)

        if config.draw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (point_x, point_y), 4, color, -1)

        if config.draw_labels:
            label = f"{track.class_name} #{track.track_id}"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if save_trails:
            history = track_histories.get(track.track_id, deque())
            if len(history) > 1:
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [points], False, color, 2)

    _draw_transition_panel_near_zones(annotated, analytics, config, zone_colors)
    _draw_status_panel(annotated, analytics, config.tracker_name, frame_idx)
    return annotated


def _draw_status_panel(frame, analytics: AnalyticsEngine, tracker_name: str, frame_idx: int) -> None:
    cv2 = _require_cv2()
    live_counts = analytics.get_live_counts()
    line_counts = live_counts["line_counts"]
    movement_counts = live_counts["movement_counts"]

    lines = [
        f"Tracker: {tracker_name}",
        f"Frame: {frame_idx}",
        "Line counts:",
    ]
    for line_name, count in sorted(line_counts.items()):
        lines.append(f"  {line_name}: {count}")

    lines.append("Movements:")
    for movement_label in ("left", "straight", "right", "unknown"):
        lines.append(f"  {movement_label}: {movement_counts.get(movement_label, 0)}")

    x = 15
    y = 25
    for text in lines:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22


def _draw_transition_panel_near_zones(
    frame,
    analytics: AnalyticsEngine,
    config: RuntimeConfig,
    zone_colors: dict[str, tuple[int, int, int]],
) -> None:
    cv2 = _require_cv2()
    transition_matrix = analytics.get_transition_matrix()
    if not transition_matrix:
        return

    zones_by_name = {zone.name: zone for zone in config.zones}
    for target_zone, source_counts in transition_matrix.items():
        zone = zones_by_name.get(target_zone)
        if zone is None:
            continue

        center_x, center_y = polygon_centroid(zone.polygon)
        y = int(center_y)
        for source_zone, count in sorted(source_counts.items()):
            color = zone_colors.get(source_zone, (255, 255, 255))
            cv2.putText(
                frame,
                f"{source_zone}: {count}",
                (int(center_x), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 22


def _draw_scene_geometry(
    frame,
    config: RuntimeConfig,
    zone_colors: dict[str, tuple[int, int, int]],
) -> None:
    cv2 = _require_cv2()

    if config.draw_zones:
        for zone in config.zones:
            polygon = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            zone_color = zone_colors[zone.name]
            cv2.polylines(frame, [polygon], True, zone_color, 2)
            anchor = tuple(polygon[0][0])
            cv2.putText(
                frame,
                zone.name,
                (int(anchor[0]), int(anchor[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone_color,
                2,
                cv2.LINE_AA,
            )

    if config.draw_lines:
        for count_line in config.count_lines:
            start_point = tuple(int(value) for value in count_line.points[0])
            end_point = tuple(int(value) for value in count_line.points[1])
            cv2.line(frame, start_point, end_point, (0, 165, 255), 2)
            midpoint = (
                int((start_point[0] + end_point[0]) / 2),
                int((start_point[1] + end_point[1]) / 2),
            )
            cv2.putText(
                frame,
                count_line.name,
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    if config.active_area is not None:
        active_area = np.array(config.active_area, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [active_area], True, (255, 255, 0), 1)


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 99, 71),
        (135, 206, 250),
        (60, 179, 113),
        (255, 215, 0),
        (186, 85, 211),
        (255, 160, 122),
        (0, 191, 255),
    )
    return palette[track_id % len(palette)]


def _build_zone_color_map(config: RuntimeConfig) -> dict[str, tuple[int, int, int]]:
    palette = (
        (255, 99, 71),
        (135, 206, 250),
        (60, 179, 113),
        (255, 215, 0),
        (186, 85, 211),
        (255, 160, 122),
        (0, 191, 255),
    )
    return {
        zone.name: palette[index % len(palette)]
        for index, zone in enumerate(config.zones)
    }
