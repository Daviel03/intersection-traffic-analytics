from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traffic_analytics.config import PROJECT_ROOT, load_runtime_config
from traffic_analytics.io_utils import ensure_dir, write_csv
from traffic_analytics.pipeline import run_pipeline

DEFAULT_SCENES = ("intersection_demo", "intersection_behnam")
DEFAULT_TRACKERS = ("bytetrack", "botsort")
MOVEMENT_LABELS = ("left", "straight", "right", "unknown")

METRICS_SUMMARY_FIELDS = [
    "scene_name",
    "tracker_name",
    "model",
    "total_line_crossings",
    "total_zone_transitions",
    "left_count",
    "straight_count",
    "right_count",
    "unknown_count",
    "unknown_movement_ratio",
    "avg_track_length",
    "short_track_ratio",
    "suspected_handoff_count",
    "duplicate_suppressed_events",
]

TRANSITION_FIELDS = [
    "scene_name",
    "tracker_name",
    "transition_name",
    "count",
]


def resolve_scene_path(scene_arg: str | Path) -> Path:
    candidate = Path(scene_arg)
    if candidate.is_absolute():
        return candidate
    if candidate.suffix in {".yaml", ".yml"}:
        return (PROJECT_ROOT / candidate).resolve()
    return (PROJECT_ROOT / "configs" / "scenes" / f"{candidate}.yaml").resolve()


def normalize_trackers(trackers: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not trackers:
        return DEFAULT_TRACKERS

    normalized = tuple(tracker.lower() for tracker in trackers)
    invalid = [tracker for tracker in normalized if tracker not in DEFAULT_TRACKERS]
    if invalid:
        raise ValueError(
            "Unsupported tracker names: " + ", ".join(sorted(set(invalid)))
        )
    return normalized


def normalize_scenes(scenes: list[str] | tuple[str, ...] | None) -> tuple[Path, ...]:
    if not scenes:
        scenes = list(DEFAULT_SCENES)
    return tuple(resolve_scene_path(scene) for scene in scenes)


def load_summary_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def get_or_create_summary(
    scene_path: str | Path,
    tracker_name: str,
    output_root: str | Path = "outputs",
    force_rerun: bool = False,
) -> dict[str, Any]:
    runtime_config = load_runtime_config(
        scene_path=scene_path,
        tracker_name=tracker_name,
        output_root=output_root,
    )
    summary_path = runtime_config.output_dir / "summary.json"
    if summary_path.exists() and not force_rerun:
        return load_summary_json(summary_path)

    result = run_pipeline(
        scene_path=scene_path,
        tracker_name=tracker_name,
        output_root=output_root,
    )
    return dict(result.summary)


def summary_to_metrics_row(summary: dict[str, Any]) -> dict[str, object]:
    movement_counts = summary.get("movement_counts", {})
    transition_counts = summary.get("transition_counts", {})
    continuity_proxies = summary.get("continuity_proxies", {})
    comparison_ready_metrics = summary.get("comparison_ready_metrics", {})
    line_counts = summary.get("line_counts", {})

    total_line_crossings = int(
        comparison_ready_metrics.get(
            "total_line_crossings",
            sum(int(value) for value in line_counts.values()),
        )
    )
    total_zone_transitions = int(
        sum(int(value) for value in transition_counts.values())
    )

    return {
        "scene_name": str(summary.get("output_name", "")),
        "tracker_name": str(summary.get("tracker_name", "")),
        "model": str(summary.get("model", "")),
        "total_line_crossings": total_line_crossings,
        "total_zone_transitions": total_zone_transitions,
        "left_count": int(movement_counts.get("left", 0)),
        "straight_count": int(movement_counts.get("straight", 0)),
        "right_count": int(movement_counts.get("right", 0)),
        "unknown_count": int(movement_counts.get("unknown", 0)),
        "unknown_movement_ratio": round(
            float(comparison_ready_metrics.get("unknown_movement_ratio", 0.0)),
            6,
        ),
        "avg_track_length": round(
            float(continuity_proxies.get("average_track_length_frames", 0.0)),
            6,
        ),
        "short_track_ratio": round(
            float(continuity_proxies.get("short_track_ratio", 0.0)),
            6,
        ),
        "suspected_handoff_count": int(
            continuity_proxies.get("suspected_id_handoff_count", 0)
        ),
        "duplicate_suppressed_events": int(
            summary.get("duplicate_suppressed_events", 0)
        ),
    }


def summary_to_transition_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    scene_name = str(summary.get("output_name", ""))
    tracker_name = str(summary.get("tracker_name", ""))
    transition_counts = summary.get("transition_counts", {})

    rows: list[dict[str, object]] = []
    for transition_name in sorted(transition_counts):
        rows.append(
            {
                "scene_name": scene_name,
                "tracker_name": tracker_name,
                "transition_name": str(transition_name),
                "count": int(transition_counts[transition_name]),
            }
        )
    return rows


def aggregate_experiment_rows(
    summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows = [summary_to_metrics_row(summary) for summary in summaries]
    transition_rows: list[dict[str, object]] = []
    for summary in summaries:
        transition_rows.extend(summary_to_transition_rows(summary))

    metric_rows.sort(key=lambda row: (str(row["scene_name"]), str(row["tracker_name"])))
    transition_rows.sort(
        key=lambda row: (
            str(row["scene_name"]),
            str(row["transition_name"]),
            str(row["tracker_name"]),
        )
    )
    return metric_rows, transition_rows


def write_experiment_outputs(
    metric_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    experiments_root: Path,
) -> dict[str, Path]:
    ensure_dir(experiments_root)
    tables_dir = experiments_root / "tables"
    ensure_dir(tables_dir)

    metrics_summary_path = experiments_root / "metrics_summary.csv"
    transition_counts_path = experiments_root / "transition_counts.csv"
    analytics_table_path = tables_dir / "analytics_table.tex"
    continuity_table_path = tables_dir / "continuity_table.tex"

    write_csv(metric_rows, METRICS_SUMMARY_FIELDS, metrics_summary_path)
    write_csv(transition_rows, TRANSITION_FIELDS, transition_counts_path)
    analytics_table_path.write_text(
        render_analytics_latex_table(metric_rows),
        encoding="utf-8",
    )
    continuity_table_path.write_text(
        render_continuity_latex_table(metric_rows),
        encoding="utf-8",
    )

    return {
        "metrics_summary_csv": metrics_summary_path,
        "transition_counts_csv": transition_counts_path,
        "analytics_table_tex": analytics_table_path,
        "continuity_table_tex": continuity_table_path,
    }


def render_analytics_latex_table(metric_rows: list[dict[str, object]]) -> str:
    headers = (
        "Scene",
        "Tracker",
        "Line Crossings",
        "Zone Transitions",
        "Left",
        "Straight",
        "Right",
        "Unknown",
        "Unknown Ratio",
    )
    rows = []
    for row in metric_rows:
        rows.append(
            (
                _latex_escape(_display_scene_name(str(row["scene_name"]))),
                _latex_escape(_display_tracker_name(str(row["tracker_name"]))),
                str(int(row["total_line_crossings"])),
                str(int(row["total_zone_transitions"])),
                str(int(row["left_count"])),
                str(int(row["straight_count"])),
                str(int(row["right_count"])),
                str(int(row["unknown_count"])),
                _format_float(float(row["unknown_movement_ratio"])),
            )
        )
    return _render_booktabs_table(
        headers=headers,
        rows=rows,
        alignment="llrrrrrrr",
        caption="Application-level traffic analytics metrics by scene and tracker.",
        label="tab:traffic_analytics_metrics",
    )


def render_continuity_latex_table(metric_rows: list[dict[str, object]]) -> str:
    headers = (
        "Scene",
        "Tracker",
        "Avg Track Length",
        "Short-Track Ratio",
        "Suspected Handoffs",
        "Duplicate Suppressed",
    )
    rows = []
    for row in metric_rows:
        rows.append(
            (
                _latex_escape(_display_scene_name(str(row["scene_name"]))),
                _latex_escape(_display_tracker_name(str(row["tracker_name"]))),
                _format_float(float(row["avg_track_length"])),
                _format_float(float(row["short_track_ratio"])),
                str(int(row["suspected_handoff_count"])),
                str(int(row["duplicate_suppressed_events"])),
            )
        )
    return _render_booktabs_table(
        headers=headers,
        rows=rows,
        alignment="llrrrr",
        caption="Tracking continuity proxy metrics by scene and tracker.",
        label="tab:tracking_continuity_metrics",
    )


def _render_booktabs_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    alignment: str,
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _display_scene_name(scene_name: str) -> str:
    return scene_name.replace("_", " ").title()


def _display_tracker_name(tracker_name: str) -> str:
    mapping = {
        "bytetrack": "ByteTrack",
        "botsort": "BoT-SORT",
    }
    return mapping.get(tracker_name.lower(), tracker_name)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    escaped = value
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped
