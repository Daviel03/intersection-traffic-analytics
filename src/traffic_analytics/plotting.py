from __future__ import annotations

import csv
from pathlib import Path

from traffic_analytics.io_utils import ensure_dir

VARIANT_ORDER = (
    "camera_bytetrack",
    "camera_botsort",
    "camera_lidar_bytetrack_fusion",
    "camera_lidar_botsort_fusion",
    "camera_lidar_fusion",
)
VARIANT_LABELS = {
    "camera_bytetrack": "Camera ByteTrack",
    "camera_botsort": "Camera BoT-SORT",
    "camera_lidar_bytetrack_fusion": "Camera+LiDAR ByteTrack",
    "camera_lidar_botsort_fusion": "Camera+LiDAR BoT-SORT",
    "camera_lidar_fusion": "Camera+LiDAR Fusion",
}
VARIANT_COLORS = {
    "camera_bytetrack": "#1f77b4",
    "camera_botsort": "#ff7f0e",
    "camera_lidar_bytetrack_fusion": "#2ca02c",
    "camera_lidar_botsort_fusion": "#d62728",
    "camera_lidar_fusion": "#9467bd",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def create_experiment_plots(
    experiments_root: Path,
    scenes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[Path]]:
    metrics_summary_path = experiments_root / "metrics_summary.csv"
    transition_counts_path = experiments_root / "transition_counts.csv"

    if not metrics_summary_path.exists():
        raise FileNotFoundError(
            f"Experiment metrics CSV not found: {metrics_summary_path}. "
            "Run scripts/run_experiments.py or scripts/run_fusion_experiments.py first."
        )
    if not transition_counts_path.exists():
        raise FileNotFoundError(
            f"Transition counts CSV not found: {transition_counts_path}. "
            "Run scripts/run_experiments.py or scripts/run_fusion_experiments.py first."
        )

    metric_rows = load_csv_rows(metrics_summary_path)
    transition_rows = load_csv_rows(transition_counts_path)
    gt_summary_path = experiments_root / "gt" / "gt_eval_summary.csv"
    gt_rows = load_csv_rows(gt_summary_path) if gt_summary_path.exists() else []

    available_scenes = sorted({row["scene_name"] for row in metric_rows})
    target_scenes = list(scenes) if scenes else available_scenes

    plots_dir = experiments_root / "plots"
    ensure_dir(plots_dir)

    output_paths: dict[str, list[Path]] = {}
    for scene_name in target_scenes:
        scene_metric_rows = [row for row in metric_rows if row["scene_name"] == scene_name]
        if not scene_metric_rows:
            continue

        output_paths[scene_name] = []
        output_paths[scene_name].extend(
            _create_standard_scene_plots(scene_name, scene_metric_rows, transition_rows, plots_dir)
        )

        scene_gt_rows = [row for row in gt_rows if row["scene_name"] == scene_name]
        if scene_gt_rows:
            output_paths[scene_name].extend(_create_gt_plots(scene_name, scene_gt_rows, plots_dir))

    return output_paths


def _create_standard_scene_plots(
    scene_name: str,
    scene_metric_rows: list[dict[str, str]],
    transition_rows: list[dict[str, str]],
    plots_dir: Path,
) -> list[Path]:
    scene_transition_rows = [row for row in transition_rows if row["scene_name"] == scene_name]

    outputs = []
    outputs.extend(
        _save_grouped_bar_chart(
            categories=("left", "straight", "right", "unknown"),
            variant_values=_scene_variant_values(
                scene_metric_rows,
                {
                    "left": "left_count",
                    "straight": "straight_count",
                    "right": "right_count",
                    "unknown": "unknown_count",
                },
            ),
            title=f"{_display_scene_name(scene_name)}: Movement Counts",
            ylabel="Count",
            output_stem=plots_dir / f"movement_counts_{scene_name}",
        )
    )
    outputs.extend(
        _save_grouped_bar_chart(
            categories=("total_line_crossings", "total_zone_transitions"),
            variant_values=_scene_variant_values(
                scene_metric_rows,
                {
                    "total_line_crossings": "total_line_crossings",
                    "total_zone_transitions": "total_zone_transitions",
                },
            ),
            title=f"{_display_scene_name(scene_name)}: Total Analytics Counts",
            ylabel="Count",
            output_stem=plots_dir / f"total_counts_{scene_name}",
        )
    )
    outputs.extend(
        _save_grouped_bar_chart(
            categories=(
                "avg_track_length",
                "short_track_ratio",
                "suspected_handoff_count",
                "duplicate_suppressed_events",
            ),
            variant_values=_scene_variant_values(
                scene_metric_rows,
                {
                    "avg_track_length": "avg_track_length",
                    "short_track_ratio": "short_track_ratio",
                    "suspected_handoff_count": "suspected_handoff_count",
                    "duplicate_suppressed_events": "duplicate_suppressed_events",
                },
            ),
            title=f"{_display_scene_name(scene_name)}: Tracking Proxy Metrics",
            ylabel="Value",
            output_stem=plots_dir / f"tracking_proxies_{scene_name}",
        )
    )

    transition_names = sorted(
        {row["transition_name"] for row in scene_transition_rows if row["transition_name"]}
    )
    if transition_names:
        outputs.extend(
            _save_grouped_bar_chart(
                categories=tuple(transition_names),
                variant_values=_transition_variant_values(
                    scene_transition_rows,
                    transition_names,
                    _variant_order(scene_metric_rows),
                ),
                title=f"{_display_scene_name(scene_name)}: Transition Counts",
                ylabel="Count",
                output_stem=plots_dir / f"transitions_{scene_name}",
                rotate_xticks=True,
            )
        )

    if any(int(row.get("fusion_enabled", "0") or 0) == 1 for row in scene_metric_rows):
        outputs.extend(
            _save_grouped_bar_chart(
                categories=(
                    "lidar_supported_track_count",
                    "lidar_unsupported_track_count",
                    "fused_confirmation_events",
                    "suppressed_camera_only_tracks",
                    "average_lidar_support_ratio",
                ),
                variant_values=_scene_variant_values(
                    scene_metric_rows,
                    {
                        "lidar_supported_track_count": "lidar_supported_track_count",
                        "lidar_unsupported_track_count": "lidar_unsupported_track_count",
                        "fused_confirmation_events": "fused_confirmation_events",
                        "suppressed_camera_only_tracks": "suppressed_camera_only_tracks",
                        "average_lidar_support_ratio": "average_lidar_support_ratio",
                    },
                ),
                title=f"{_display_scene_name(scene_name)}: Fusion Diagnostics",
                ylabel="Value",
                output_stem=plots_dir / f"fusion_diagnostics_{scene_name}",
            )
        )
    return outputs


def _create_gt_plots(
    scene_name: str,
    gt_rows: list[dict[str, str]],
    plots_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    subset_names = sorted({row["subset_name"] for row in gt_rows if row.get("subset_name")})
    for subset_name in subset_names:
        subset_rows = [row for row in gt_rows if row.get("subset_name") == subset_name]
        variant_values = _gt_variant_values(subset_rows)
        outputs.extend(
            _save_grouped_bar_chart(
                categories=("HOTA", "IDF1", "MOTA"),
                variant_values=variant_values,
                title=(
                    f"{_display_scene_name(scene_name)}: "
                    f"GT Tracking Metrics ({_display_scene_name(subset_name)})"
                ),
                ylabel="Score",
                output_stem=plots_dir / f"gt_metrics_{scene_name}_{subset_name}",
            )
        )
    return outputs


def _scene_variant_values(
    scene_metric_rows: list[dict[str, str]],
    category_to_column: dict[str, str],
) -> dict[str, list[float]]:
    row_by_variant = {_variant_name(row): row for row in scene_metric_rows}
    values = {variant: [] for variant in _variant_order(scene_metric_rows)}
    for variant_name in values:
        row = row_by_variant.get(variant_name, {})
        for category in category_to_column:
            values[variant_name].append(_to_float(row.get(category_to_column[category])))
    return values


def _transition_variant_values(
    scene_transition_rows: list[dict[str, str]],
    transition_names: list[str],
    variant_names: tuple[str, ...],
) -> dict[str, list[float]]:
    values = {variant: [] for variant in variant_names}
    lookup: dict[tuple[str, str], float] = {}
    for row in scene_transition_rows:
        lookup[(_variant_name(row), row["transition_name"])] = _to_float(row["count"])

    for variant_name in values:
        for transition_name in transition_names:
            values[variant_name].append(lookup.get((variant_name, transition_name), 0.0))
    return values


def _gt_variant_values(
    subset_rows: list[dict[str, str]],
) -> dict[str, list[float]]:
    values = {variant: [] for variant in _variant_order(subset_rows)}
    row_by_variant = {_variant_name(row): row for row in subset_rows}
    for variant_name in values:
        row = row_by_variant.get(variant_name)
        if row is None:
            values[variant_name] = [0.0, 0.0, 0.0]
            continue
        values[variant_name] = [
            _to_float(row.get("HOTA")),
            _to_float(row.get("IDF1")),
            _to_float(row.get("MOTA")),
        ]
    return values


def _save_grouped_bar_chart(
    categories: tuple[str, ...],
    variant_values: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_stem: Path,
    rotate_xticks: bool = False,
) -> list[Path]:
    plt = _require_pyplot()
    x_positions = list(range(len(categories)))
    variant_names = list(variant_values.keys())
    variant_count = max(1, len(variant_names))
    width = min(0.75 / variant_count, 0.28)
    figure_width = max(7.0, 1.2 * len(categories) + 4.0)
    fig, ax = plt.subplots(figsize=(figure_width, 4.8))

    origin = -(width * (variant_count - 1) / 2.0)
    for index, variant_name in enumerate(variant_names):
        values = variant_values.get(variant_name, [0.0] * len(categories))
        bar_positions = [x + origin + index * width for x in x_positions]
        ax.bar(
            bar_positions,
            values,
            width=width,
            label=_display_variant_name(variant_name),
            color=VARIANT_COLORS.get(variant_name, "#7f7f7f"),
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_display_category_name(category) for category in categories],
        rotation=20 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
    )
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=200 if suffix == ".png" else None, bbox_inches="tight")
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def _variant_name(row: dict[str, str]) -> str:
    explicit = str(row.get("system_variant", "")).strip().lower()
    if explicit:
        return explicit
    tracker_name = str(row.get("tracker_name", "")).strip().lower()
    if tracker_name:
        return f"camera_{tracker_name}"
    return "camera_unknown"


def _variant_order(rows: list[dict[str, str]]) -> tuple[str, ...]:
    present = {_variant_name(row) for row in rows}
    ordered = [variant for variant in VARIANT_ORDER if variant in present]
    extras = sorted(present - set(ordered))
    return tuple(ordered + extras)


def _display_variant_name(variant_name: str) -> str:
    return VARIANT_LABELS.get(variant_name, variant_name.replace("_", " ").title())


def _display_scene_name(scene_name: str) -> str:
    return scene_name.replace("_", " ").title()


def _display_category_name(category: str) -> str:
    return category.replace("_", " ").replace("->", " -> ").title()


def _to_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _require_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
