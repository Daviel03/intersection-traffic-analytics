from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from traffic_analytics.geometry import bottom_center, distance
from traffic_analytics.io_utils import write_csv

BBox = tuple[float, float, float, float]
Point = tuple[float, float]

LIDAR_EVIDENCE_FIELDS = [
    "frame_idx",
    "object_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "range_m",
    "support_score",
]


@dataclass(frozen=True)
class LidarEvidence:
    frame_idx: int
    object_id: str
    center: Point
    support_score: float
    bbox: BBox | None = None
    range_m: float | None = None

    def to_csv_row(self) -> dict[str, object]:
        row = {
            "frame_idx": self.frame_idx,
            "object_id": self.object_id,
            "center_x": round(self.center[0], 6),
            "center_y": round(self.center[1], 6),
            "range_m": "" if self.range_m is None else round(self.range_m, 6),
            "support_score": round(self.support_score, 6),
        }
        if self.bbox is None:
            row.update({"x1": "", "y1": "", "x2": "", "y2": ""})
        else:
            row.update(
                {
                    "x1": round(self.bbox[0], 6),
                    "y1": round(self.bbox[1], 6),
                    "x2": round(self.bbox[2], 6),
                    "y2": round(self.bbox[3], 6),
                }
            )
        return row


def resolve_default_lidar_evidence_path(
    scene_name: str,
    lidar_root: str | Path = "data/lidar",
) -> Path:
    return Path(lidar_root) / scene_name / "evidence.csv"


def load_lidar_evidence_csv(path: Path) -> list[LidarEvidence]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    evidence: list[LidarEvidence] = []
    for row in rows:
        bbox = _parse_optional_bbox(row)
        center = _parse_center(row, bbox)
        range_raw = row.get("range_m")
        evidence.append(
            LidarEvidence(
                frame_idx=int(row["frame_idx"]),
                object_id=str(row.get("object_id", "")),
                center=center,
                support_score=float(row.get("support_score") or 0.0),
                bbox=bbox,
                range_m=None if range_raw in (None, "") else float(range_raw),
            )
        )
    return evidence


def write_lidar_evidence_csv(records: list[LidarEvidence], path: Path) -> None:
    write_csv(
        [record.to_csv_row() for record in records],
        LIDAR_EVIDENCE_FIELDS,
        path,
    )


def group_lidar_evidence_by_frame(
    records: list[LidarEvidence],
) -> dict[int, list[LidarEvidence]]:
    grouped: dict[int, list[LidarEvidence]] = defaultdict(list)
    for record in records:
        grouped[record.frame_idx].append(record)
    return dict(grouped)


def generate_mock_lidar_evidence_from_track_rows(
    track_rows: list[dict[str, object]],
    min_motion_px: float = 20.0,
    min_track_frames: int = 3,
    support_score: float = 0.9,
) -> list[LidarEvidence]:
    rows_by_track: dict[int, list[dict[str, object]]] = defaultdict(list)
    max_y2 = 0.0
    for row in track_rows:
        track_id = int(row["track_id"])
        rows_by_track[track_id].append(row)
        max_y2 = max(max_y2, float(row["y2"]))

    image_height_estimate = max(max_y2, 1.0)
    evidence: list[LidarEvidence] = []
    for track_id, rows in rows_by_track.items():
        ordered_rows = sorted(rows, key=lambda row: int(row["frame_idx"]))
        if len(ordered_rows) < min_track_frames:
            continue

        start_bbox = _row_bbox(ordered_rows[0])
        end_bbox = _row_bbox(ordered_rows[-1])
        motion_px = distance(bottom_center(start_bbox), bottom_center(end_bbox))
        if motion_px < min_motion_px:
            continue

        per_track_support = min(1.0, support_score + min(0.1, motion_px / 500.0))
        for row in ordered_rows:
            bbox = _row_bbox(row)
            point = (float(row["point_x"]), float(row["point_y"]))
            evidence.append(
                LidarEvidence(
                    frame_idx=int(row["frame_idx"]),
                    object_id=f"mock_{track_id}",
                    center=point,
                    support_score=per_track_support,
                    bbox=bbox,
                    range_m=_estimate_mock_range_m(point[1], image_height_estimate),
                )
            )

    evidence.sort(key=lambda item: (item.frame_idx, item.object_id))
    return evidence


def _parse_optional_bbox(row: dict[str, str]) -> BBox | None:
    keys = ("x1", "y1", "x2", "y2")
    values = [row.get(key, "") for key in keys]
    if any(value in (None, "") for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _parse_center(row: dict[str, str], bbox: BBox | None) -> Point:
    if row.get("center_x") not in (None, "") and row.get("center_y") not in (None, ""):
        return (float(row["center_x"]), float(row["center_y"]))
    if bbox is not None:
        return bottom_center(bbox)
    raise ValueError(f"LiDAR evidence row is missing both center and bbox fields: {row}")


def _row_bbox(row: dict[str, object]) -> BBox:
    return (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )


def _estimate_mock_range_m(point_y: float, image_height_estimate: float) -> float:
    normalized = min(max(point_y / image_height_estimate, 0.0), 1.0)
    return round(5.0 + (1.0 - normalized) * 45.0, 3)
