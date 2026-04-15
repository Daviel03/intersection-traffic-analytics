from __future__ import annotations

import csv
from pathlib import Path

from traffic_analytics.io_utils import ensure_dir

TRACKER_ORDER = ("bytetrack", "botsort")
TRACKER_LABELS = {
    "bytetrack": "ByteTrack",
    "botsort": "BoT-SORT",
}
TRACKER_COLORS = {
    "bytetrack": "#1f77b4",
    "botsort": "#ff7f0e",
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
            "Run scripts/run_experiments.py first."
        )
    if not transition_counts_path.exists():
        raise FileNotFoundError(
            f"Transition counts CSV not found: {transition_counts_path}. "
            "Run scripts/run_experiments.py first."
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
            output_paths[scene_name].extend(
                _create_gt_plots(scene_name, scene_gt_rows, plots_dir)
            )

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
            scene_name=scene_name,
            categories=("left", "straight", "right", "unknown"),
            tracker_values=_scene_tracker_values(
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
            scene_name=scene_name,
            categories=("total_line_crossings", "total_zone_transitions"),
            tracker_values=_scene_tracker_values(
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
            scene_name=scene_name,
            categories=(
                "avg_track_length",
                "short_track_ratio",
                "suspected_handoff_count",
                "duplicate_suppressed_events",
            ),
            tracker_values=_scene_tracker_values(
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
                scene_name=scene_name,
                categories=tuple(transition_names),
                tracker_values=_transition_tracker_values(scene_transition_rows, transition_names),
                title=f"{_display_scene_name(scene_name)}: Transition Counts",
                ylabel="Count",
                output_stem=plots_dir / f"transitions_{scene_name}",
                rotate_xticks=True,
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
        tracker_values = {tracker: [] for tracker in TRACKER_ORDER}
        per_tracker = {row["tracker_name"].lower(): row for row in subset_rows}
        for tracker_name in TRACKER_ORDER:
            row = per_tracker.get(tracker_name)
            if row is None:
                tracker_values[tracker_name] = [0.0, 0.0, 0.0]
                continue
            tracker_values[tracker_name] = [
                _to_float(row.get("HOTA")),
                _to_float(row.get("IDF1")),
                _to_float(row.get("MOTA")),
            ]

        outputs.extend(
            _save_grouped_bar_chart(
                scene_name=scene_name,
                categories=("HOTA", "IDF1", "MOTA"),
                tracker_values=tracker_values,
                title=(
                    f"{_display_scene_name(scene_name)}: "
                    f"GT Tracking Metrics ({_display_scene_name(subset_name)})"
                ),
                ylabel="Score",
                output_stem=plots_dir / f"gt_metrics_{scene_name}_{subset_name}",
            )
        )
    return outputs


def _scene_tracker_values(
    scene_metric_rows: list[dict[str, str]],
    category_to_column: dict[str, str],
) -> dict[str, list[float]]:
    row_by_tracker = {
        row["tracker_name"].lower(): row for row in scene_metric_rows
    }
    values = {tracker: [] for tracker in TRACKER_ORDER}
    for tracker_name in TRACKER_ORDER:
        row = row_by_tracker.get(tracker_name, {})
        for category in category_to_column:
            values[tracker_name].append(_to_float(row.get(category_to_column[category])))
    return values


def _transition_tracker_values(
    scene_transition_rows: list[dict[str, str]],
    transition_names: list[str],
) -> dict[str, list[float]]:
    values = {tracker: [] for tracker in TRACKER_ORDER}
    lookup: dict[tuple[str, str], float] = {}
    for row in scene_transition_rows:
        lookup[(row["tracker_name"].lower(), row["transition_name"])] = _to_float(row["count"])

    for tracker_name in TRACKER_ORDER:
        for transition_name in transition_names:
            values[tracker_name].append(lookup.get((tracker_name, transition_name), 0.0))
    return values


def _save_grouped_bar_chart(
    scene_name: str,
    categories: tuple[str, ...],
    tracker_values: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_stem: Path,
    rotate_xticks: bool = False,
) -> list[Path]:
    plt = _require_pyplot()
    x_positions = list(range(len(categories)))
    width = 0.35
    figure_width = max(7.0, 1.2 * len(categories) + 4.0)
    fig, ax = plt.subplots(figsize=(figure_width, 4.8))

    offsets = {
        "bytetrack": -width / 2,
        "botsort": width / 2,
    }
    for tracker_name in TRACKER_ORDER:
        values = tracker_values.get(tracker_name, [0.0] * len(categories))
        bar_positions = [x + offsets[tracker_name] for x in x_positions]
        ax.bar(
            bar_positions,
            values,
            width=width,
            label=TRACKER_LABELS[tracker_name],
            color=TRACKER_COLORS[tracker_name],
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
