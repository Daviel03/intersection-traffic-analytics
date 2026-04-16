from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from traffic_analytics.analytics import AnalyticsEngine, EventRecord
from traffic_analytics.config import RuntimeConfig
from traffic_analytics.geometry import bbox_iou, distance
from traffic_analytics.lidar import LidarEvidence, group_lidar_evidence_by_frame
from traffic_analytics.tracker_backend import TrackedObject

FUSION_MATCH_FIELDS = [
    "frame_idx",
    "track_id",
    "class_name",
    "camera_confidence",
    "lidar_supported",
    "lidar_object_id",
    "lidar_support_score",
    "lidar_range_m",
    "match_iou",
    "center_distance_px",
    "fused_confidence",
    "track_confirmed",
]


@dataclass(frozen=True)
class FusionSettings:
    min_bbox_iou: float = 0.05
    max_center_distance_px: float = 80.0
    min_support_score: float = 0.3
    min_track_support_ratio: float = 0.25
    min_supported_frames: int = 2
    min_track_frames: int = 3


def load_track_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fuse_track_rows_with_lidar(
    track_rows: list[dict[str, object]] | list[dict[str, str]],
    lidar_records: list[LidarEvidence],
    settings: FusionSettings | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    settings = settings or FusionSettings()
    lidar_by_frame = group_lidar_evidence_by_frame(lidar_records)

    augmented_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    support_counts: dict[int, int] = defaultdict(int)
    total_counts: dict[int, int] = defaultdict(int)

    sorted_rows = sorted(
        [{key: value for key, value in row.items()} for row in track_rows],
        key=lambda row: (int(row["frame_idx"]), int(row["track_id"])),
    )
    for row in sorted_rows:
        track = TrackedObject.from_csv_row(row)
        evidence, match_iou, center_distance_px = _match_track_to_lidar(
            track=track,
            lidar_records=lidar_by_frame.get(track.frame_idx, []),
            settings=settings,
        )
        lidar_supported = evidence is not None
        support_score = evidence.support_score if evidence is not None else 0.0
        range_m = evidence.range_m if evidence is not None else None
        fused_confidence = _compute_fused_confidence(
            camera_confidence=track.confidence,
            lidar_supported=lidar_supported,
            support_score=support_score,
        )

        total_counts[track.track_id] += 1
        if lidar_supported:
            support_counts[track.track_id] += 1

        augmented_row = dict(row)
        augmented_row.update(
            {
                "lidar_supported": int(lidar_supported),
                "lidar_support_score": round(support_score, 6),
                "lidar_range_m": "" if range_m is None else round(range_m, 6),
                "fused_confidence": round(fused_confidence, 6),
            }
        )
        augmented_rows.append(augmented_row)
        match_rows.append(
            {
                "frame_idx": track.frame_idx,
                "track_id": track.track_id,
                "class_name": track.class_name,
                "camera_confidence": round(track.confidence, 6),
                "lidar_supported": int(lidar_supported),
                "lidar_object_id": "" if evidence is None else evidence.object_id,
                "lidar_support_score": round(support_score, 6),
                "lidar_range_m": "" if range_m is None else round(range_m, 6),
                "match_iou": round(match_iou, 6),
                "center_distance_px": round(center_distance_px, 6),
                "fused_confidence": round(fused_confidence, 6),
                "track_confirmed": 0,
            }
        )

    confirmed_track_ids, support_ratios = _resolve_confirmed_tracks(
        total_counts=total_counts,
        support_counts=support_counts,
        settings=settings,
    )
    suppressed_track_ids = set(total_counts) - confirmed_track_ids

    fused_rows = [
        row
        for row in augmented_rows
        if int(row["track_id"]) in confirmed_track_ids
    ]
    for row in match_rows:
        row["track_confirmed"] = int(int(row["track_id"]) in confirmed_track_ids)

    diagnostics = {
        "lidar_supported_track_count": len(confirmed_track_ids),
        "lidar_unsupported_track_count": len(suppressed_track_ids),
        "fused_confirmation_events": len(confirmed_track_ids),
        "suppressed_camera_only_tracks": len(suppressed_track_ids),
        "average_lidar_support_ratio": round(
            mean(support_ratios.values()) if support_ratios else 0.0,
            6,
        ),
        "confirmed_track_ids": sorted(confirmed_track_ids),
        "suppressed_track_ids": sorted(suppressed_track_ids),
        "track_support_ratios": {
            str(track_id): round(support_ratio, 6)
            for track_id, support_ratio in sorted(support_ratios.items())
        },
    }
    return fused_rows, match_rows, diagnostics


def replay_analytics_from_track_rows(
    track_rows: list[dict[str, object]] | list[dict[str, str]],
    config: RuntimeConfig,
) -> tuple[dict[str, object], list[EventRecord]]:
    analytics = AnalyticsEngine(
        count_lines=config.count_lines,
        zones=config.zones,
        movement_map=config.movement_map,
    )
    rows_by_frame: dict[int, list[TrackedObject]] = defaultdict(list)
    for row in track_rows:
        track = TrackedObject.from_csv_row(row)
        rows_by_frame[track.frame_idx].append(track)

    for frame_idx in sorted(rows_by_frame):
        frame_tracks = sorted(
            rows_by_frame[frame_idx],
            key=lambda track: track.track_id,
        )
        analytics_tracks = [
            track for track in frame_tracks if track.class_name in config.analytics_classes
        ]
        timestamp_sec = frame_tracks[0].timestamp_sec if frame_tracks else 0.0
        analytics.process_tracks(analytics_tracks, frame_idx, timestamp_sec)

    summary = analytics.finalize()
    return summary, list(analytics.events)


def _match_track_to_lidar(
    track: TrackedObject,
    lidar_records: list[LidarEvidence],
    settings: FusionSettings,
) -> tuple[LidarEvidence | None, float, float]:
    best_record: LidarEvidence | None = None
    best_score = float("-inf")
    best_iou = 0.0
    best_center_distance = float("inf")

    for record in lidar_records:
        if record.support_score < settings.min_support_score:
            continue

        current_iou = bbox_iou(track.bbox, record.bbox) if record.bbox is not None else 0.0
        current_center_distance = distance(track.point, record.center)
        if (
            current_iou < settings.min_bbox_iou
            and current_center_distance > settings.max_center_distance_px
        ):
            continue

        distance_term = max(
            0.0,
            1.0 - min(current_center_distance, settings.max_center_distance_px)
            / max(settings.max_center_distance_px, 1.0),
        )
        match_score = current_iou + 0.5 * distance_term + 0.25 * record.support_score
        if match_score > best_score:
            best_record = record
            best_score = match_score
            best_iou = current_iou
            best_center_distance = current_center_distance

    if best_record is None:
        return None, 0.0, float("inf")
    return best_record, best_iou, best_center_distance


def _resolve_confirmed_tracks(
    total_counts: dict[int, int],
    support_counts: dict[int, int],
    settings: FusionSettings,
) -> tuple[set[int], dict[int, float]]:
    confirmed: set[int] = set()
    support_ratios: dict[int, float] = {}
    for track_id, total_frames in total_counts.items():
        supported_frames = support_counts.get(track_id, 0)
        support_ratio = (
            supported_frames / total_frames if total_frames else 0.0
        )
        support_ratios[track_id] = support_ratio
        if total_frames < settings.min_track_frames:
            continue
        if supported_frames >= settings.min_supported_frames or support_ratio >= settings.min_track_support_ratio:
            confirmed.add(track_id)
    return confirmed, support_ratios


def _compute_fused_confidence(
    camera_confidence: float,
    lidar_supported: bool,
    support_score: float,
) -> float:
    if not lidar_supported:
        return max(0.0, camera_confidence * 0.9)
    return min(1.0, camera_confidence + 0.15 * support_score)
