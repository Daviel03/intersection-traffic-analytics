from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from traffic_analytics.config import RuntimeConfig
from traffic_analytics.geometry import distance


def summarize_track_rows(track_rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    per_track: dict[int, dict[str, object]] = {}

    for row in sorted(track_rows, key=lambda item: (int(item["track_id"]), int(item["frame_idx"]))):
        track_id = int(row["track_id"])
        point = (float(row["point_x"]), float(row["point_y"]))
        if track_id not in per_track:
            per_track[track_id] = {
                "track_id": track_id,
                "class_name": str(row["class_name"]),
                "start_frame": int(row["frame_idx"]),
                "end_frame": int(row["frame_idx"]),
                "start_point": point,
                "end_point": point,
                "frame_count": 1,
            }
            continue

        track_summary = per_track[track_id]
        track_summary["end_frame"] = int(row["frame_idx"])
        track_summary["end_point"] = point
        track_summary["frame_count"] = int(track_summary["frame_count"]) + 1

    return per_track


def detect_suspected_id_handoffs(
    per_track: dict[int, dict[str, object]],
    max_gap_frames: int,
    max_distance_px: float,
) -> int:
    ended_tracks = sorted(per_track.values(), key=lambda item: int(item["end_frame"]))
    started_tracks = sorted(per_track.values(), key=lambda item: int(item["start_frame"]))

    used_successors: set[int] = set()
    handoff_count = 0

    for ended_track in ended_tracks:
        best_candidate: tuple[int, float, int] | None = None
        for candidate in started_tracks:
            candidate_id = int(candidate["track_id"])
            if candidate_id == int(ended_track["track_id"]) or candidate_id in used_successors:
                continue
            if candidate["class_name"] != ended_track["class_name"]:
                continue

            frame_gap = int(candidate["start_frame"]) - int(ended_track["end_frame"])
            if frame_gap < 1 or frame_gap > max_gap_frames:
                continue

            point_gap = distance(
                ended_track["end_point"],  # type: ignore[arg-type]
                candidate["start_point"],  # type: ignore[arg-type]
            )
            if point_gap > max_distance_px:
                continue

            ranking = (frame_gap, point_gap, candidate_id)
            if best_candidate is None or ranking < best_candidate:
                best_candidate = ranking

        if best_candidate is None:
            continue

        used_successors.add(best_candidate[2])
        handoff_count += 1

    return handoff_count


def build_run_summary(
    config: RuntimeConfig,
    analytics_summary: dict[str, object],
    track_rows: list[dict[str, object]],
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    per_track = summarize_track_rows(track_rows)
    frame_lengths = [int(item["frame_count"]) for item in per_track.values()]
    unique_track_count = len(per_track)
    average_track_length_frames = mean(frame_lengths) if frame_lengths else 0.0
    short_track_count = sum(
        1
        for length in frame_lengths
        if length < config.short_track_threshold_frames
    )
    short_track_ratio = (
        short_track_count / unique_track_count if unique_track_count else 0.0
    )
    suspected_id_handoff_count = detect_suspected_id_handoffs(
        per_track,
        max_gap_frames=config.handoff_max_gap_frames,
        max_distance_px=config.handoff_max_distance_px,
    )
    class_track_counts = build_class_track_counts(per_track)
    class_detection_counts = build_class_detection_counts(track_rows)
    class_average_track_length_frames = build_class_average_track_lengths(per_track)
    for class_name in config.target_classes:
        class_track_counts.setdefault(class_name, 0)
        class_detection_counts.setdefault(class_name, 0)
        class_average_track_length_frames.setdefault(class_name, 0.0)

    line_counts = dict(analytics_summary["line_counts"])
    zone_entry_counts = dict(analytics_summary["zone_entry_counts"])
    zone_exit_counts = dict(analytics_summary["zone_exit_counts"])
    movement_counts = dict(analytics_summary["movement_counts"])
    transition_counts = dict(analytics_summary["transition_counts"])
    transition_matrix = dict(analytics_summary.get("transition_matrix", {}))
    analytic_track_count = int(analytics_summary["analytic_track_count"])
    unknown_movement_ratio = (
        float(movement_counts.get("unknown", 0)) / analytic_track_count
        if analytic_track_count
        else 0.0
    )

    return {
        "scene_path": str(config.scene_path),
        "video_path": str(config.video_path),
        "output_name": config.output_name,
        "tracker_name": config.tracker_name,
        "model": config.model,
        "target_classes": list(config.target_classes),
        "analytics_classes": list(config.analytics_classes),
        "frame_count": frame_count,
        "fps": round(fps, 6),
        "class_track_counts": class_track_counts,
        "class_detection_counts": class_detection_counts,
        "class_average_track_length_frames": class_average_track_length_frames,
        "line_counts": line_counts,
        "zone_entry_counts": zone_entry_counts,
        "zone_exit_counts": zone_exit_counts,
        "movement_counts": movement_counts,
        "transition_counts": transition_counts,
        "transition_matrix": transition_matrix,
        "duplicate_suppressed_events": int(analytics_summary["duplicate_suppressed_events"]),
        "unknown_track_ids": list(analytics_summary["unknown_track_ids"]),
        "continuity_proxies": {
            "unique_track_count": unique_track_count,
            "average_track_length_frames": round(average_track_length_frames, 6),
            "short_track_ratio": round(short_track_ratio, 6),
            "suspected_id_handoff_count": suspected_id_handoff_count,
        },
        "comparison_ready_metrics": {
            "total_line_crossings": int(sum(line_counts.values())),
            "unknown_movement_ratio": round(unknown_movement_ratio, 6),
        },
    }


def build_class_track_counts(
    per_track: dict[int, dict[str, object]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for track_summary in per_track.values():
        counts[str(track_summary["class_name"])] += 1
    return dict(counts)


def build_class_detection_counts(
    track_rows: list[dict[str, object]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in track_rows:
        counts[str(row["class_name"])] += 1
    return dict(counts)


def build_class_average_track_lengths(
    per_track: dict[int, dict[str, object]]
) -> dict[str, float]:
    lengths_by_class: dict[str, list[int]] = defaultdict(list)
    for track_summary in per_track.values():
        lengths_by_class[str(track_summary["class_name"])].append(
            int(track_summary["frame_count"])
        )
    return {
        class_name: round(mean(lengths), 6)
        for class_name, lengths in lengths_by_class.items()
        if lengths
    }


def build_comparison_rows(
    summaries: dict[str, dict[str, Any]]
) -> list[dict[str, object]]:
    bytetrack_summary = summaries["bytetrack"]
    botsort_summary = summaries["botsort"]
    rows: list[dict[str, object]] = []

    def add_row(layer: str, metric: str, key: str, bytetrack_value: float, botsort_value: float) -> None:
        rows.append(
            {
                "layer": layer,
                "metric": metric,
                "key": key,
                "bytetrack": bytetrack_value,
                "botsort": botsort_value,
                "delta_botsort_minus_bytetrack": round(botsort_value - bytetrack_value, 6),
            }
        )

    continuity_metrics = (
        "unique_track_count",
        "average_track_length_frames",
        "short_track_ratio",
        "suspected_id_handoff_count",
    )
    for metric in continuity_metrics:
        add_row(
            "continuity",
            metric,
            "",
            float(bytetrack_summary["continuity_proxies"][metric]),
            float(botsort_summary["continuity_proxies"][metric]),
        )

    add_row(
        "continuity",
        "duplicate_suppressed_events",
        "",
        float(bytetrack_summary["duplicate_suppressed_events"]),
        float(botsort_summary["duplicate_suppressed_events"]),
    )

    add_row(
        "downstream",
        "total_line_crossings",
        "",
        float(bytetrack_summary["comparison_ready_metrics"]["total_line_crossings"]),
        float(botsort_summary["comparison_ready_metrics"]["total_line_crossings"]),
    )
    add_row(
        "downstream",
        "unknown_movement_ratio",
        "",
        float(bytetrack_summary["comparison_ready_metrics"]["unknown_movement_ratio"]),
        float(botsort_summary["comparison_ready_metrics"]["unknown_movement_ratio"]),
    )

    for metric_name in (
        "line_counts",
        "movement_counts",
        "transition_counts",
        "class_track_counts",
        "class_detection_counts",
    ):
        keys = sorted(
            set(bytetrack_summary.get(metric_name, {}).keys())
            | set(botsort_summary.get(metric_name, {}).keys())
        )
        for key in keys:
            add_row(
                "downstream",
                metric_name,
                key,
                float(bytetrack_summary.get(metric_name, {}).get(key, 0)),
                float(botsort_summary.get(metric_name, {}).get(key, 0)),
            )

    return rows


def build_comparison_payload(
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, object]],
) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["layer"])].append(row)

    return {
        "trackers": summaries,
        "rows": rows,
        "rows_by_layer": dict(grouped_rows),
    }


def build_quick_comparison_payload(
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, object]],
) -> dict[str, Any]:
    bytetrack_summary = summaries["bytetrack"]
    botsort_summary = summaries["botsort"]
    top_differences = sorted(
        (
            row
            for row in rows
            if row["metric"] not in {"unknown_movement_ratio"}
        ),
        key=lambda row: abs(float(row["delta_botsort_minus_bytetrack"])),
        reverse=True,
    )[:8]

    class_names = sorted(
        set(bytetrack_summary.get("target_classes", []))
        | set(botsort_summary.get("target_classes", []))
        | set(bytetrack_summary.get("class_track_counts", {}).keys())
        | set(botsort_summary.get("class_track_counts", {}).keys())
    )
    class_breakdown = []
    for class_name in class_names:
        bt_tracks = int(bytetrack_summary.get("class_track_counts", {}).get(class_name, 0))
        bs_tracks = int(botsort_summary.get("class_track_counts", {}).get(class_name, 0))
        bt_rows = int(bytetrack_summary.get("class_detection_counts", {}).get(class_name, 0))
        bs_rows = int(botsort_summary.get("class_detection_counts", {}).get(class_name, 0))
        class_breakdown.append(
            {
                "class_name": class_name,
                "unique_tracks": {
                    "bytetrack": bt_tracks,
                    "botsort": bs_tracks,
                    "delta_botsort_minus_bytetrack": bs_tracks - bt_tracks,
                },
                "frame_detections": {
                    "bytetrack": bt_rows,
                    "botsort": bs_rows,
                    "delta_botsort_minus_bytetrack": bs_rows - bt_rows,
                },
            }
        )

    return {
        "scene": bytetrack_summary["output_name"],
        "model": bytetrack_summary["model"],
        "target_classes": bytetrack_summary.get("target_classes", []),
        "analytics_classes": bytetrack_summary.get("analytics_classes", []),
        "headlines": {
            "total_line_crossings": {
                "bytetrack": bytetrack_summary["comparison_ready_metrics"]["total_line_crossings"],
                "botsort": botsort_summary["comparison_ready_metrics"]["total_line_crossings"],
            },
            "unknown_movement_ratio": {
                "bytetrack": bytetrack_summary["comparison_ready_metrics"]["unknown_movement_ratio"],
                "botsort": botsort_summary["comparison_ready_metrics"]["unknown_movement_ratio"],
            },
            "unique_track_count": {
                "bytetrack": bytetrack_summary["continuity_proxies"]["unique_track_count"],
                "botsort": botsort_summary["continuity_proxies"]["unique_track_count"],
            },
            "suspected_id_handoff_count": {
                "bytetrack": bytetrack_summary["continuity_proxies"]["suspected_id_handoff_count"],
                "botsort": botsort_summary["continuity_proxies"]["suspected_id_handoff_count"],
            },
        },
        "class_breakdown": class_breakdown,
        "movement_counts": {
            "bytetrack": bytetrack_summary.get("movement_counts", {}),
            "botsort": botsort_summary.get("movement_counts", {}),
        },
        "top_differences": top_differences,
    }


def render_quick_comparison_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Quick Comparison: {payload['scene']}")
    lines.append("")
    lines.append(f"- Model: `{payload['model']}`")
    lines.append(f"- Target classes: `{', '.join(payload.get('target_classes', []))}`")
    lines.append(f"- Analytics classes: `{', '.join(payload.get('analytics_classes', []))}`")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append("| Metric | ByteTrack | BoT-SORT |")
    lines.append("| --- | ---: | ---: |")
    for metric_name, values in payload["headlines"].items():
        lines.append(
            f"| {metric_name} | {values['bytetrack']} | {values['botsort']} |"
        )
    lines.append("")
    lines.append("## Class Breakdown")
    lines.append("")
    lines.append("| Class | ByteTrack Tracks | BoT-SORT Tracks | ByteTrack Frame Detections | BoT-SORT Frame Detections |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for item in payload["class_breakdown"]:
        lines.append(
            f"| {item['class_name']} | {item['unique_tracks']['bytetrack']} | "
            f"{item['unique_tracks']['botsort']} | {item['frame_detections']['bytetrack']} | "
            f"{item['frame_detections']['botsort']} |"
        )
    lines.append("")
    lines.append("## Movement Counts")
    lines.append("")
    lines.append("| Movement | ByteTrack | BoT-SORT |")
    lines.append("| --- | ---: | ---: |")
    movement_keys = sorted(
        set(payload["movement_counts"]["bytetrack"].keys())
        | set(payload["movement_counts"]["botsort"].keys())
    )
    for movement_label in movement_keys:
        lines.append(
            f"| {movement_label} | "
            f"{payload['movement_counts']['bytetrack'].get(movement_label, 0)} | "
            f"{payload['movement_counts']['botsort'].get(movement_label, 0)} |"
        )
    lines.append("")
    lines.append("## Largest Differences")
    lines.append("")
    lines.append("| Layer | Metric | Key | Delta (BoT-SORT - ByteTrack) |")
    lines.append("| --- | --- | --- | ---: |")
    for row in payload["top_differences"]:
        lines.append(
            f"| {row['layer']} | {row['metric']} | {row['key'] or '-'} | {row['delta_botsort_minus_bytetrack']} |"
        )
    lines.append("")
    return "\n".join(lines)
