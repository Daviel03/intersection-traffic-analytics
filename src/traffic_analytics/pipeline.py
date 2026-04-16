from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from traffic_analytics.analytics import AnalyticsEngine
from traffic_analytics.config import RuntimeConfig, load_runtime_config
from traffic_analytics.evaluation import build_run_summary
from traffic_analytics.geometry import point_in_polygon
from traffic_analytics.io_utils import (
    _require_cv2,
    create_video_writer,
    ensure_dir,
    write_csv,
    write_json,
)
from traffic_analytics.tracker_backend import TrackedObject, UltralyticsTrackerBackend
from traffic_analytics.visualization import render_annotated_frame

TRACK_FIELDS = [
    "frame_idx",
    "timestamp_sec",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "point_x",
    "point_y",
    "lidar_supported",
    "lidar_support_score",
    "lidar_range_m",
    "fused_confidence",
]

EVENT_FIELDS = [
    "event_type",
    "track_id",
    "frame_idx",
    "timestamp_sec",
    "target_name",
    "source_zone",
    "target_zone",
    "movement_label",
    "suppressed_duplicate",
]


@dataclass(frozen=True)
class PipelineResult:
    tracker_name: str
    output_dir: Path
    annotated_video_path: Path
    tracks_path: Path
    events_path: Path
    summary_path: Path
    summary: dict[str, object]


def run_pipeline(
    scene_path: str | Path,
    tracker_name: str,
    output_root: str | Path | None = None,
    save_trails: bool = False,
) -> PipelineResult:
    config = load_runtime_config(
        scene_path=scene_path,
        tracker_name=tracker_name,
        output_root=output_root,
    )
    _validate_input_video(config)
    cv2 = _require_cv2()

    ensure_dir(config.output_dir)

    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    backend = UltralyticsTrackerBackend(
        model_path=config.model,
        tracker_config_path=config.tracker_config_path,
        target_classes=config.target_classes,
        confidence=config.confidence,
        iou=config.iou,
        device=config.device,
    )
    analytics = AnalyticsEngine(
        count_lines=config.count_lines,
        zones=config.zones,
        movement_map=config.movement_map,
    )

    annotated_video_path = config.output_dir / "annotated.mp4"
    writer = create_video_writer(
        annotated_video_path,
        fps=fps,
        frame_size=(frame_width, frame_height),
    )

    track_rows: list[dict[str, object]] = []
    track_histories: dict[int, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=config.trail_length)
    )

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_sec = frame_idx / fps if fps else float(frame_idx)
            tracked_objects = backend.track_frame(frame, frame_idx, timestamp_sec)
            active_tracks = _filter_tracks_by_active_area(tracked_objects, config)
            analytics_tracks = _filter_tracks_for_analytics(active_tracks, config)

            analytics.process_tracks(analytics_tracks, frame_idx, timestamp_sec)

            for track in active_tracks:
                track_rows.append(track.to_csv_row())
                track_histories[track.track_id].append(track.point)

            annotated_frame = render_annotated_frame(
                frame=frame,
                tracks=active_tracks,
                analytics=analytics,
                config=config,
                frame_idx=frame_idx,
                track_histories=track_histories,
                save_trails=save_trails,
            )
            writer.write(annotated_frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    analytics_summary = analytics.finalize()

    tracks_path = config.output_dir / "tracks.csv"
    events_path = config.output_dir / "events.csv"
    summary_path = config.output_dir / "summary.json"

    write_csv(track_rows, TRACK_FIELDS, tracks_path)
    write_csv(
        [event.to_csv_row() for event in analytics.events],
        EVENT_FIELDS,
        events_path,
    )

    summary = build_run_summary(
        config=config,
        analytics_summary=analytics_summary,
        track_rows=track_rows,
        frame_count=frame_idx,
        fps=fps,
    )
    write_json(summary, summary_path)

    return PipelineResult(
        tracker_name=config.tracker_name,
        output_dir=config.output_dir,
        annotated_video_path=annotated_video_path,
        tracks_path=tracks_path,
        events_path=events_path,
        summary_path=summary_path,
        summary=summary,
    )


def _filter_tracks_by_active_area(
    tracked_objects: list[TrackedObject],
    config: RuntimeConfig,
) -> list[TrackedObject]:
    if config.active_area is None:
        return tracked_objects
    return [
        track
        for track in tracked_objects
        if point_in_polygon(track.point, config.active_area)
    ]


def _filter_tracks_for_analytics(
    tracked_objects: list[TrackedObject],
    config: RuntimeConfig,
) -> list[TrackedObject]:
    allowed_classes = set(config.analytics_classes)
    return [track for track in tracked_objects if track.class_name in allowed_classes]


def _validate_input_video(config: RuntimeConfig) -> None:
    if not config.video_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {config.video_path}. "
            "Place your local clip in data/ and update the scene YAML if needed."
        )
