from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MOVEMENTS = {"left", "straight", "right"}

Point = tuple[float, float]


@dataclass(frozen=True)
class CountLineConfig:
    name: str
    points: tuple[Point, Point]


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    polygon: tuple[Point, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    scene_path: Path
    video_path: Path
    lidar_evidence_path: Path | None
    output_name: str
    output_dir: Path
    comparison_dir: Path
    model: str
    target_classes: tuple[str, ...]
    analytics_classes: tuple[str, ...]
    count_lines: tuple[CountLineConfig, ...]
    zones: tuple[ZoneConfig, ...]
    movement_map: dict[str, dict[str, str]]
    active_area: tuple[Point, ...] | None
    tracker_name: str
    tracker_config_path: Path
    confidence: float
    iou: float
    device: str | None
    short_track_threshold_frames: int
    handoff_max_gap_frames: int
    handoff_max_distance_px: float
    trail_length: int
    draw_boxes: bool
    draw_labels: bool
    draw_lines: bool
    draw_zones: bool


def load_runtime_config(
    scene_path: str | Path,
    tracker_name: str,
    output_root: str | Path | None = None,
) -> RuntimeConfig:
    tracker_name = tracker_name.lower()
    if tracker_name not in {"bytetrack", "botsort"}:
        raise ValueError(f"Unsupported tracker '{tracker_name}'.")

    defaults_path = PROJECT_ROOT / "configs" / "default.yaml"
    resolved_scene_path = _resolve_path(scene_path)
    defaults = _load_yaml(defaults_path)
    scene = _load_yaml(resolved_scene_path)

    required_keys = (
        "video_path",
        "output_name",
        "target_classes",
        "count_lines",
        "zones",
        "movement_map",
    )
    missing = [key for key in required_keys if key not in scene]
    if missing:
        raise ValueError(
            f"Scene config is missing required keys: {', '.join(sorted(missing))}."
        )

    tracking_defaults = defaults.get("tracking", {})
    evaluation_defaults = defaults.get("evaluation", {})
    visualization_defaults = defaults.get("visualization", {})

    video_path = _resolve_path(scene["video_path"])
    lidar_evidence_path = (
        _resolve_path(scene["lidar_evidence_path"])
        if "lidar_evidence_path" in scene
        else None
    )
    output_name = str(scene["output_name"])
    resolved_output_root = _resolve_path(output_root or "outputs")
    output_base = resolved_output_root / output_name
    target_classes = tuple(str(item) for item in scene["target_classes"])
    analytics_classes = tuple(
        str(item) for item in scene.get("analytics_classes", scene["target_classes"])
    )
    invalid_analytics_classes = [
        class_name for class_name in analytics_classes if class_name not in target_classes
    ]
    if invalid_analytics_classes:
        raise ValueError(
            "analytics_classes must be a subset of target_classes. Invalid values: "
            + ", ".join(invalid_analytics_classes)
        )

    return RuntimeConfig(
        project_root=PROJECT_ROOT,
        scene_path=resolved_scene_path,
        video_path=video_path,
        lidar_evidence_path=lidar_evidence_path,
        output_name=output_name,
        output_dir=output_base / tracker_name,
        comparison_dir=output_base / "comparison",
        model=str(scene.get("model", defaults.get("model", "yolov8n.pt"))),
        target_classes=target_classes,
        analytics_classes=analytics_classes,
        count_lines=_parse_count_lines(scene["count_lines"]),
        zones=_parse_zones(scene["zones"]),
        movement_map=_validate_movement_map(scene["movement_map"]),
        active_area=_parse_polygon(scene.get("active_area")) if "active_area" in scene else None,
        tracker_name=tracker_name,
        tracker_config_path=PROJECT_ROOT / "configs" / "trackers" / f"{tracker_name}.yaml",
        confidence=float(tracking_defaults.get("confidence", 0.25)),
        iou=float(tracking_defaults.get("iou", 0.45)),
        device=_normalize_optional_string(tracking_defaults.get("device")),
        short_track_threshold_frames=int(
            evaluation_defaults.get("short_track_threshold_frames", 10)
        ),
        handoff_max_gap_frames=int(evaluation_defaults.get("handoff_max_gap_frames", 8)),
        handoff_max_distance_px=float(
            evaluation_defaults.get("handoff_max_distance_px", 80.0)
        ),
        trail_length=int(visualization_defaults.get("trail_length", 30)),
        draw_boxes=bool(visualization_defaults.get("draw_boxes", True)),
        draw_labels=bool(visualization_defaults.get("draw_labels", True)),
        draw_lines=bool(visualization_defaults.get("draw_lines", True)),
        draw_zones=bool(visualization_defaults.get("draw_zones", True)),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}.")
    return data


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(os.path.expandvars(str(value)))
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _normalize_optional_string(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)


def _parse_point(raw_point: Any) -> Point:
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        raise ValueError(f"Expected [x, y] point, got: {raw_point!r}")
    return (float(raw_point[0]), float(raw_point[1]))


def _parse_polygon(raw_polygon: Any) -> tuple[Point, ...]:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        raise ValueError("Polygon must contain at least three points.")
    return tuple(_parse_point(point) for point in raw_polygon)


def _parse_count_lines(raw_lines: Any) -> tuple[CountLineConfig, ...]:
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("count_lines must be a non-empty list.")
    parsed: list[CountLineConfig] = []
    for item in raw_lines:
        if not isinstance(item, dict) or "name" not in item or "points" not in item:
            raise ValueError("Each count line needs a name and points field.")
        points = item["points"]
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("Each count line must have exactly two points.")
        parsed.append(
            CountLineConfig(
                name=str(item["name"]),
                points=(_parse_point(points[0]), _parse_point(points[1])),
            )
        )
    return tuple(parsed)


def _parse_zones(raw_zones: Any) -> tuple[ZoneConfig, ...]:
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ValueError("zones must be a non-empty list.")
    parsed: list[ZoneConfig] = []
    for item in raw_zones:
        if not isinstance(item, dict) or "name" not in item or "polygon" not in item:
            raise ValueError("Each zone needs a name and polygon field.")
        parsed.append(
            ZoneConfig(
                name=str(item["name"]),
                polygon=_parse_polygon(item["polygon"]),
            )
        )
    return tuple(parsed)


def _validate_movement_map(raw_movement_map: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw_movement_map, dict) or not raw_movement_map:
        raise ValueError("movement_map must be a non-empty mapping.")

    normalized: dict[str, dict[str, str]] = {}
    for source_zone, targets in raw_movement_map.items():
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"movement_map[{source_zone!r}] must be a mapping.")
        normalized_targets: dict[str, str] = {}
        for target_zone, label in targets.items():
            normalized_label = str(label).lower()
            if normalized_label not in ALLOWED_MOVEMENTS:
                raise ValueError(
                    f"Invalid movement label '{label}' for {source_zone}->{target_zone}."
                )
            normalized_targets[str(target_zone)] = normalized_label
        normalized[str(source_zone)] = normalized_targets
    return normalized
