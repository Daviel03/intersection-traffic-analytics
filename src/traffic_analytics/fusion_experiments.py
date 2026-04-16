from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_analytics.config import load_runtime_config
from traffic_analytics.evaluation import build_run_summary
from traffic_analytics.experiments import (
    aggregate_experiment_rows,
    fusion_system_variant_name,
    get_or_create_summary,
    load_summary_json,
    normalize_scenes,
    normalize_trackers,
    write_experiment_outputs,
)
from traffic_analytics.fusion import (
    FUSION_MATCH_FIELDS,
    FusionSettings,
    fuse_track_rows_with_lidar,
    load_track_rows,
    replay_analytics_from_track_rows,
)
from traffic_analytics.lidar import (
    generate_mock_lidar_evidence_from_track_rows,
    load_lidar_evidence_csv,
    resolve_default_lidar_evidence_path,
    write_lidar_evidence_csv,
)
from traffic_analytics.io_utils import ensure_dir, write_csv, write_json
from traffic_analytics.pipeline import EVENT_FIELDS, TRACK_FIELDS


def run_fusion_variant(
    scene_path: str | Path,
    tracker_name: str,
    output_root: str | Path = "outputs",
    lidar_root: str | Path = "data/lidar",
    lidar_evidence_path: str | Path | None = None,
    allow_mock_lidar: bool = True,
    force_rerun: bool = False,
    fusion_settings: FusionSettings | None = None,
) -> dict[str, Any]:
    camera_summary = get_or_create_summary(
        scene_path=scene_path,
        tracker_name=tracker_name,
        output_root=output_root,
        force_rerun=force_rerun,
    )
    config = load_runtime_config(
        scene_path=scene_path,
        tracker_name=tracker_name,
        output_root=output_root,
    )
    system_variant = fusion_system_variant_name(tracker_name)
    fused_output_dir = config.output_dir.parent / system_variant
    summary_path = fused_output_dir / "summary.json"
    if summary_path.exists() and not force_rerun:
        return load_summary_json(summary_path)

    camera_tracks_path = config.output_dir / "tracks.csv"
    if not camera_tracks_path.exists():
        raise FileNotFoundError(
            f"Camera track CSV not found: {camera_tracks_path}. Run the scene pipeline first."
        )
    camera_track_rows = load_track_rows(camera_tracks_path)

    resolved_evidence_path = _resolve_lidar_evidence_path(
        config_lidar_evidence_path=config.lidar_evidence_path,
        scene_name=config.output_name,
        lidar_root=lidar_root,
        lidar_evidence_path=lidar_evidence_path,
    )
    if resolved_evidence_path.exists():
        lidar_records = load_lidar_evidence_csv(resolved_evidence_path)
        lidar_evidence_source = "precomputed"
    elif allow_mock_lidar:
        lidar_records = generate_mock_lidar_evidence_from_track_rows(camera_track_rows)
        lidar_evidence_source = "mock_motion_proxy"
    else:
        raise FileNotFoundError(
            f"LiDAR evidence not found: {resolved_evidence_path}. "
            "Provide precomputed evidence or allow mock evidence generation."
        )

    fused_track_rows, match_rows, fusion_diagnostics = fuse_track_rows_with_lidar(
        track_rows=camera_track_rows,
        lidar_records=lidar_records,
        settings=fusion_settings,
    )
    analytics_summary, events = replay_analytics_from_track_rows(
        track_rows=fused_track_rows,
        config=config,
    )
    summary = build_run_summary(
        config=config,
        analytics_summary=analytics_summary,
        track_rows=fused_track_rows,
        frame_count=int(camera_summary.get("frame_count", 0)),
        fps=float(camera_summary.get("fps", 0.0)),
        system_variant=system_variant,
        fusion_enabled=True,
        fusion_diagnostics=fusion_diagnostics,
    )
    summary["lidar_evidence_source"] = lidar_evidence_source
    summary["lidar_evidence_path"] = str(resolved_evidence_path)

    ensure_dir(fused_output_dir)
    write_csv(fused_track_rows, TRACK_FIELDS, fused_output_dir / "tracks.csv")
    write_csv(
        [event.to_csv_row() for event in events],
        EVENT_FIELDS,
        fused_output_dir / "events.csv",
    )
    write_csv(match_rows, FUSION_MATCH_FIELDS, fused_output_dir / "fusion_matches.csv")
    write_lidar_evidence_csv(lidar_records, fused_output_dir / "lidar_evidence.csv")
    write_json(summary, summary_path)
    return summary


def collect_experiment_summaries(
    scenes: list[str] | tuple[str, ...] | None = None,
    trackers: list[str] | tuple[str, ...] | None = None,
    fusion_trackers: list[str] | tuple[str, ...] | None = None,
    output_root: str | Path = "outputs",
    lidar_root: str | Path = "data/lidar",
    allow_mock_lidar: bool = True,
    force_rerun: bool = False,
    fusion_settings: FusionSettings | None = None,
) -> list[dict[str, Any]]:
    scene_paths = normalize_scenes(scenes)
    tracker_names = normalize_trackers(trackers)
    fusion_tracker_names = normalize_trackers(fusion_trackers) if fusion_trackers else ("bytetrack",)

    summaries: list[dict[str, Any]] = []
    for scene_path in scene_paths:
        for tracker_name in tracker_names:
            summaries.append(
                get_or_create_summary(
                    scene_path=scene_path,
                    tracker_name=tracker_name,
                    output_root=output_root,
                    force_rerun=force_rerun,
                )
            )
        for tracker_name in fusion_tracker_names:
            summaries.append(
                run_fusion_variant(
                    scene_path=scene_path,
                    tracker_name=tracker_name,
                    output_root=output_root,
                    lidar_root=lidar_root,
                    allow_mock_lidar=allow_mock_lidar,
                    force_rerun=force_rerun,
                    fusion_settings=fusion_settings,
                )
            )
    return summaries


def run_and_write_fusion_experiments(
    scenes: list[str] | tuple[str, ...] | None = None,
    trackers: list[str] | tuple[str, ...] | None = None,
    fusion_trackers: list[str] | tuple[str, ...] | None = None,
    output_root: str | Path = "outputs",
    lidar_root: str | Path = "data/lidar",
    allow_mock_lidar: bool = True,
    force_rerun: bool = False,
    fusion_settings: FusionSettings | None = None,
) -> dict[str, Path]:
    summaries = collect_experiment_summaries(
        scenes=scenes,
        trackers=trackers,
        fusion_trackers=fusion_trackers,
        output_root=output_root,
        lidar_root=lidar_root,
        allow_mock_lidar=allow_mock_lidar,
        force_rerun=force_rerun,
        fusion_settings=fusion_settings,
    )
    metric_rows, transition_rows = aggregate_experiment_rows(summaries)
    experiments_root = (Path(output_root) / "experiments").resolve()
    return write_experiment_outputs(metric_rows, transition_rows, experiments_root)


def _resolve_lidar_evidence_path(
    config_lidar_evidence_path: Path | None,
    scene_name: str,
    lidar_root: str | Path,
    lidar_evidence_path: str | Path | None,
) -> Path:
    if lidar_evidence_path is not None:
        return Path(lidar_evidence_path).resolve()
    if config_lidar_evidence_path is not None:
        return config_lidar_evidence_path.resolve()
    return resolve_default_lidar_evidence_path(scene_name=scene_name, lidar_root=lidar_root).resolve()
