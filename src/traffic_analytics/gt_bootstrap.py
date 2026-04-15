from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from traffic_analytics.config import PROJECT_ROOT
from traffic_analytics.experiments import normalize_trackers, resolve_scene_path
from traffic_analytics.gt_eval import load_csv_rows
from traffic_analytics.io_utils import ensure_dir, write_csv

GT_TRACK_FIELDS = ["frame_idx", "track_id", "class_name", "x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class BootstrapResult:
    scene_name: str
    subset_name: str
    frame_start: int
    frame_end: int
    classes: tuple[str, ...]
    gt_row_count: int
    gt_track_count: int
    subset_yaml_path: Path
    gt_tracks_path: Path


def bootstrap_consensus_subset(
    scene_name: str,
    subset_name: str,
    frame_start: int,
    frame_end: int,
    classes: tuple[str, ...],
    tracker_names: tuple[str, ...],
    ground_truth_root: Path,
    output_root: Path,
    iou_threshold: float = 0.5,
    description: str | None = None,
) -> BootstrapResult:
    tracker_names = normalize_trackers(list(tracker_names))
    if len(tracker_names) != 2:
        raise ValueError("Consensus bootstrap currently supports exactly two trackers.")
    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start.")
    if not classes:
        raise ValueError("classes must be a non-empty tuple of class names.")

    predictions_by_tracker = {}
    for tracker_name in tracker_names:
        tracks_path = output_root / scene_name / tracker_name / "tracks.csv"
        if not tracks_path.exists():
            raise FileNotFoundError(
                f"Prediction tracks CSV not found: {tracks_path}. Run the scene pipeline first."
            )
        predictions_by_tracker[tracker_name] = filter_tracker_rows(
            load_csv_rows(tracks_path),
            frame_start=frame_start,
            frame_end=frame_end,
            classes=classes,
        )

    gt_rows = build_consensus_rows(
        predictions_by_tracker=predictions_by_tracker,
        tracker_names=tracker_names,
        iou_threshold=iou_threshold,
    )
    if not gt_rows:
        raise RuntimeError(
            "Consensus bootstrap did not produce any GT rows. "
            "Try a different frame range or a lower IoU threshold."
        )

    scene_path = resolve_scene_path(scene_name)
    video_path = _load_scene_video_path(scene_path)
    subset_dir = ground_truth_root / scene_name / subset_name
    ensure_dir(subset_dir)

    subset_payload = {
        "scene_name": scene_name,
        "video_path": video_path,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "classes": list(classes),
        "description": description
        or (
            f"Consensus-seeded long continuous chunk for {scene_name}. "
            f"Frames {frame_start}-{frame_end} were bootstrapped from "
            f"{tracker_names[0]} and {tracker_names[1]} overlap on classes "
            f"{', '.join(classes)} and should be manually reviewed before final paper claims."
        ),
    }
    subset_yaml_path = subset_dir / "subset.yaml"
    subset_yaml_path.write_text(
        yaml.safe_dump(subset_payload, sort_keys=False),
        encoding="utf-8",
    )

    gt_tracks_path = subset_dir / "gt_tracks.csv"
    write_csv(gt_rows, GT_TRACK_FIELDS, gt_tracks_path)

    return BootstrapResult(
        scene_name=scene_name,
        subset_name=subset_name,
        frame_start=frame_start,
        frame_end=frame_end,
        classes=classes,
        gt_row_count=len(gt_rows),
        gt_track_count=len({int(row["track_id"]) for row in gt_rows}),
        subset_yaml_path=subset_yaml_path,
        gt_tracks_path=gt_tracks_path,
    )


def filter_tracker_rows(
    rows: list[dict[str, str]],
    frame_start: int,
    frame_end: int,
    classes: tuple[str, ...],
) -> list[dict[str, str]]:
    allowed_classes = set(classes)
    return [
        row
        for row in rows
        if frame_start <= int(row["frame_idx"]) <= frame_end
        and row["class_name"] in allowed_classes
    ]


def build_consensus_rows(
    predictions_by_tracker: dict[str, list[dict[str, str]]],
    tracker_names: tuple[str, str],
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    tracker_a, tracker_b = tracker_names
    frame_rows = {
        tracker_name: _group_rows_by_frame(rows)
        for tracker_name, rows in predictions_by_tracker.items()
    }

    union_find = _UnionFind()
    first_seen_frame: dict[tuple[str, str], int] = {}
    all_frames = sorted(
        {
            frame_idx
            for tracker_name in tracker_names
            for frame_idx in frame_rows.get(tracker_name, {})
        }
    )

    for frame_idx in all_frames:
        rows_a = frame_rows.get(tracker_a, {}).get(frame_idx, [])
        rows_b = frame_rows.get(tracker_b, {}).get(frame_idx, [])
        for row_a, row_b in _greedy_match_rows(rows_a, rows_b, iou_threshold):
            key_a = (tracker_a, str(row_a["track_id"]))
            key_b = (tracker_b, str(row_b["track_id"]))
            union_find.add(key_a)
            union_find.add(key_b)
            union_find.union(key_a, key_b)
            first_seen_frame.setdefault(key_a, frame_idx)
            first_seen_frame.setdefault(key_b, frame_idx)

    components: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    component_first_seen: dict[tuple[str, str], int] = {}
    for key in union_find.parents:
        root = union_find.find(key)
        components[root].append(key)
        component_first_seen[root] = min(
            component_first_seen.get(root, first_seen_frame.get(key, 10**9)),
            first_seen_frame.get(key, 10**9),
        )

    ordered_roots = sorted(
        components,
        key=lambda root: (
            component_first_seen[root],
            sorted(components[root]),
        ),
    )
    key_to_canonical = {
        key: canonical_id
        for canonical_id, root in enumerate(ordered_roots, start=1)
        for key in components[root]
    }

    grouped_rows: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for tracker_name in tracker_names:
        for frame_idx, rows in frame_rows.get(tracker_name, {}).items():
            for row in rows:
                key = (tracker_name, str(row["track_id"]))
                canonical_id = key_to_canonical.get(key)
                if canonical_id is None:
                    continue
                grouped_rows[(frame_idx, canonical_id, row["class_name"])].append(row)

    consensus_rows: list[dict[str, object]] = []
    for (frame_idx, canonical_id, class_name), rows in sorted(grouped_rows.items()):
        consensus_rows.append(
            {
                "frame_idx": frame_idx,
                "track_id": canonical_id,
                "class_name": class_name,
                "x1": round(sum(float(row["x1"]) for row in rows) / len(rows), 3),
                "y1": round(sum(float(row["y1"]) for row in rows) / len(rows), 3),
                "x2": round(sum(float(row["x2"]) for row in rows) / len(rows), 3),
                "y2": round(sum(float(row["y2"]) for row in rows) / len(rows), 3),
            }
        )
    return consensus_rows


def _group_rows_by_frame(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_idx"])].append(row)
    return grouped


def _greedy_match_rows(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    iou_threshold: float,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    candidates: list[tuple[float, int, int]] = []
    for index_a, row_a in enumerate(rows_a):
        for index_b, row_b in enumerate(rows_b):
            if row_a["class_name"] != row_b["class_name"]:
                continue
            score = _bbox_iou(row_a, row_b)
            if score >= iou_threshold:
                candidates.append((score, index_a, index_b))

    matches: list[tuple[dict[str, str], dict[str, str]]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, index_a, index_b in sorted(candidates, reverse=True):
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append((rows_a[index_a], rows_b[index_b]))
    return matches


def _bbox_iou(row_a: dict[str, str], row_b: dict[str, str]) -> float:
    ax1, ay1, ax2, ay2 = (float(row_a[key]) for key in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(row_b[key]) for key in ("x1", "y1", "x2", "y2"))

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = area_a + area_b - intersection
    if denominator <= 0.0:
        return 0.0
    return intersection / denominator


def _load_scene_video_path(scene_path: Path) -> str:
    with scene_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    video_path = payload.get("video_path")
    if not video_path:
        raise ValueError(f"Scene config does not define video_path: {scene_path}")
    return str(video_path)


class _UnionFind:
    def __init__(self) -> None:
        self.parents: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, key: tuple[str, str]) -> None:
        self.parents.setdefault(key, key)

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        parent = self.parents[key]
        if parent != key:
            self.parents[key] = self.find(parent)
        return self.parents[key]

    def union(self, key_a: tuple[str, str], key_b: tuple[str, str]) -> None:
        root_a = self.find(key_a)
        root_b = self.find(key_b)
        if root_a == root_b:
            return
        if root_a < root_b:
            self.parents[root_b] = root_a
        else:
            self.parents[root_a] = root_b
