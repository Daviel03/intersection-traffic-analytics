from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from traffic_analytics.config import PROJECT_ROOT
from traffic_analytics.io_utils import ensure_dir, write_csv

GT_SUMMARY_FIELDS = [
    "scene_name",
    "subset_name",
    "tracker_name",
    "frame_start",
    "frame_end",
    "num_gt_frames",
    "num_gt_tracks",
    "HOTA",
    "IDF1",
    "MOTA",
    "IDSW",
    "FP",
    "FN",
]

FILTER_CHECK_FIELDS = [
    "scene_name",
    "subset_name",
    "tracker_name",
    "frame_start",
    "frame_end",
    "classes",
    "gt_frame_min",
    "gt_frame_max",
    "pred_frame_min",
    "pred_frame_max",
    "gt_row_count",
    "pred_row_count",
    "gt_track_count",
    "pred_track_count",
    "gt_classes_seen",
    "pred_classes_seen",
]


@dataclass(frozen=True)
class GroundTruthSubset:
    scene_name: str
    subset_name: str
    video_path: Path
    frame_start: int
    frame_end: int
    classes: tuple[str, ...]
    gt_tracks_path: Path
    description: str | None = None


def discover_ground_truth_subsets(
    ground_truth_root: Path | None = None,
) -> list[tuple[str, str]]:
    root = ground_truth_root or (PROJECT_ROOT / "data" / "ground_truth")
    if not root.exists():
        return []

    discovered: list[tuple[str, str]] = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for subset_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            if (subset_dir / "subset.yaml").exists() and (subset_dir / "gt_tracks.csv").exists():
                discovered.append((scene_dir.name, subset_dir.name))
    return discovered


def load_ground_truth_subset(
    scene_name: str,
    subset_name: str,
    ground_truth_root: Path | None = None,
) -> GroundTruthSubset:
    root = ground_truth_root or (PROJECT_ROOT / "data" / "ground_truth")
    subset_dir = root / scene_name / subset_name
    subset_path = subset_dir / "subset.yaml"
    gt_tracks_path = subset_dir / "gt_tracks.csv"

    if not subset_path.exists():
        raise FileNotFoundError(f"GT subset config not found: {subset_path}")
    if not gt_tracks_path.exists():
        raise FileNotFoundError(f"GT tracks CSV not found: {gt_tracks_path}")

    with subset_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {subset_path}")

    required_keys = ("scene_name", "video_path", "frame_start", "frame_end", "classes")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f"GT subset is missing required keys: {', '.join(sorted(missing))}"
        )

    resolved_video_path = Path(payload["video_path"])
    if not resolved_video_path.is_absolute():
        resolved_video_path = (PROJECT_ROOT / resolved_video_path).resolve()

    classes = tuple(str(class_name) for class_name in payload["classes"])
    if not classes:
        raise ValueError("GT subset classes must be a non-empty list.")

    frame_start = int(payload["frame_start"])
    frame_end = int(payload["frame_end"])
    if frame_end < frame_start:
        raise ValueError("GT subset frame_end must be greater than or equal to frame_start.")

    return GroundTruthSubset(
        scene_name=str(payload["scene_name"]),
        subset_name=subset_name,
        video_path=resolved_video_path,
        frame_start=frame_start,
        frame_end=frame_end,
        classes=classes,
        gt_tracks_path=gt_tracks_path,
        description=_normalize_optional_string(payload.get("description")),
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_ground_truth_rows(
    rows: list[dict[str, str]],
    subset: GroundTruthSubset,
) -> list[dict[str, str]]:
    return _filter_rows_by_subset(rows, subset, require_bbox=True)


def filter_prediction_rows(
    rows: list[dict[str, str]],
    subset: GroundTruthSubset,
) -> list[dict[str, str]]:
    return _filter_rows_by_subset(rows, subset, require_bbox=True)


def build_filter_check_row(
    subset: GroundTruthSubset,
    tracker_name: str,
    gt_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "scene_name": subset.scene_name,
        "subset_name": subset.subset_name,
        "tracker_name": tracker_name,
        "frame_start": subset.frame_start,
        "frame_end": subset.frame_end,
        "classes": ",".join(subset.classes),
        "gt_frame_min": _frame_min(gt_rows),
        "gt_frame_max": _frame_max(gt_rows),
        "pred_frame_min": _frame_min(prediction_rows),
        "pred_frame_max": _frame_max(prediction_rows),
        "gt_row_count": len(gt_rows),
        "pred_row_count": len(prediction_rows),
        "gt_track_count": len({row["track_id"] for row in gt_rows}),
        "pred_track_count": len({row["track_id"] for row in prediction_rows}),
        "gt_classes_seen": ",".join(sorted({row["class_name"] for row in gt_rows})),
        "pred_classes_seen": ",".join(sorted({row["class_name"] for row in prediction_rows})),
    }


def prepare_trackeval_workspace(
    subset: GroundTruthSubset,
    gt_rows: list[dict[str, str]],
    predictions_by_tracker: dict[str, list[dict[str, str]]],
    workspace_root: Path,
) -> dict[str, Path]:
    seq_name = f"{subset.scene_name}_{subset.subset_name}"
    gt_root = workspace_root / "gt"
    trackers_root = workspace_root / "trackers"
    results_root = workspace_root / "results"
    seqmap_path = workspace_root / "seqmap.txt"

    ensure_dir(gt_root)
    ensure_dir(trackers_root)
    ensure_dir(results_root)

    frame_count = subset.frame_end - subset.frame_start + 1
    fps, width, height = _probe_video_info(subset.video_path)

    gt_sequence_dir = gt_root / seq_name
    ensure_dir(gt_sequence_dir / "gt")
    _write_seqinfo(
        path=gt_sequence_dir / "seqinfo.ini",
        seq_name=seq_name,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
    _write_seqmap(seqmap_path, seq_name)
    _write_lines(
        gt_sequence_dir / "gt" / "gt.txt",
        _convert_gt_rows_to_mot_lines(gt_rows, subset.frame_start),
    )

    for tracker_name, prediction_rows in predictions_by_tracker.items():
        tracker_dir = trackers_root / tracker_name / "data"
        ensure_dir(tracker_dir)
        _write_lines(
            tracker_dir / f"{seq_name}.txt",
            _convert_prediction_rows_to_mot_lines(prediction_rows, subset.frame_start),
        )

    return {
        "workspace_root": workspace_root,
        "gt_root": gt_root,
        "trackers_root": trackers_root,
        "results_root": results_root,
        "seqmap_path": seqmap_path,
    }


def evaluate_with_trackeval(
    subset: GroundTruthSubset,
    tracker_names: tuple[str, ...],
    filtered_gt_rows: list[dict[str, str]],
    workspace_paths: dict[str, Path],
    trackeval_root: Path,
) -> list[dict[str, object]]:
    trackeval = _import_trackeval(trackeval_root)

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "DISPLAY_LESS_PROGRESS": True,
            "TIME_PROGRESS": False,
            "PLOT_CURVES": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "BREAK_ON_ERROR": True,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(workspace_paths["gt_root"]),
            "TRACKERS_FOLDER": str(workspace_paths["trackers_root"]),
            "OUTPUT_FOLDER": str(workspace_paths["results_root"]),
            "TRACKERS_TO_EVAL": list(tracker_names),
            "TRACKER_DISPLAY_NAMES": [_display_tracker_name(name) for name in tracker_names],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "CUSTOM",
            "SPLIT_TO_EVAL": "all",
            "SEQMAP_FILE": str(workspace_paths["seqmap_path"]),
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "INPUT_AS_ZIP": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "PRINT_CONFIG": False,
        }
    )

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

    evaluator = trackeval.Evaluator(evaluator_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = []
    for metric_class in (
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
    ):
        metric = metric_class(metrics_config)
        if metric.get_name() in metrics_config["METRICS"]:
            metrics.append(metric)

    output_res, output_msg = evaluator.evaluate([dataset], metrics)
    return extract_trackeval_summary_rows(
        subset=subset,
        tracker_names=tracker_names,
        filtered_gt_rows=filtered_gt_rows,
        output_res=output_res,
        output_msg=output_msg,
    )


def extract_trackeval_summary_rows(
    subset: GroundTruthSubset,
    tracker_names: tuple[str, ...],
    filtered_gt_rows: list[dict[str, str]],
    output_res: dict[str, Any],
    output_msg: dict[str, Any],
) -> list[dict[str, object]]:
    dataset_name = next(iter(output_res.keys()))
    num_gt_frames = len({int(row["frame_idx"]) for row in filtered_gt_rows})
    num_gt_tracks = len({str(row["track_id"]) for row in filtered_gt_rows})

    rows: list[dict[str, object]] = []
    for tracker_name in tracker_names:
        tracker_output = output_res.get(dataset_name, {}).get(tracker_name)
        if tracker_output is None:
            message = output_msg.get(dataset_name, {}).get(tracker_name, "Unknown TrackEval error")
            raise RuntimeError(f"TrackEval did not return results for {tracker_name}: {message}")

        combined = tracker_output["COMBINED_SEQ"]["pedestrian"]
        hota_metrics = combined["HOTA"]
        clear_metrics = combined["CLEAR"]
        identity_metrics = combined["Identity"]

        rows.append(
            {
                "scene_name": subset.scene_name,
                "subset_name": subset.subset_name,
                "tracker_name": tracker_name,
                "frame_start": subset.frame_start,
                "frame_end": subset.frame_end,
                "num_gt_frames": num_gt_frames,
                "num_gt_tracks": num_gt_tracks,
                "HOTA": _format_trackeval_percentage(hota_metrics["HOTA"]),
                "IDF1": _format_trackeval_percentage(identity_metrics["IDF1"]),
                "MOTA": _format_trackeval_percentage(clear_metrics["MOTA"]),
                "IDSW": int(round(float(clear_metrics["IDSW"]))),
                "FP": int(round(float(clear_metrics["CLR_FP"]))),
                "FN": int(round(float(clear_metrics["CLR_FN"]))),
            }
        )
    return rows


def render_gt_latex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{GT-backed tracking metrics on labeled subsets.}",
        "\\label{tab:gt_tracking_metrics}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Scene & Subset & Tracker & HOTA & IDF1 & MOTA & IDSW & FP & FN & GT Tracks \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                (
                    _latex_escape(_display_scene_name(str(row["scene_name"]))),
                    _latex_escape(str(row["subset_name"])),
                    _latex_escape(_display_tracker_name(str(row["tracker_name"]))),
                    _format_float(float(row["HOTA"])),
                    _format_float(float(row["IDF1"])),
                    _format_float(float(row["MOTA"])),
                    str(int(row["IDSW"])),
                    str(int(row["FP"])),
                    str(int(row["FN"])),
                    str(int(row["num_gt_tracks"])),
                )
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_gt_outputs(
    gt_rows: list[dict[str, object]],
    filter_check_rows: list[dict[str, object]],
    gt_root: Path,
) -> dict[str, Path]:
    ensure_dir(gt_root)
    gt_summary_path = gt_root / "gt_eval_summary.csv"
    gt_table_path = gt_root / "gt_eval_table.tex"
    filter_checks_path = gt_root / "filter_checks.csv"

    merged_gt_rows = _merge_existing_rows(
        gt_summary_path,
        gt_rows,
        key_fields=("scene_name", "subset_name", "tracker_name"),
    )
    merged_filter_rows = _merge_existing_rows(
        filter_checks_path,
        filter_check_rows,
        key_fields=("scene_name", "subset_name", "tracker_name"),
    )

    write_csv(merged_gt_rows, GT_SUMMARY_FIELDS, gt_summary_path)
    gt_table_path.write_text(render_gt_latex_table(merged_gt_rows), encoding="utf-8")
    write_csv(merged_filter_rows, FILTER_CHECK_FIELDS, filter_checks_path)

    return {
        "gt_summary_csv": gt_summary_path,
        "gt_table_tex": gt_table_path,
        "filter_checks_csv": filter_checks_path,
    }


def run_subset_evaluation(
    scene_name: str,
    subset_name: str,
    tracker_names: tuple[str, ...],
    trackeval_root: Path,
    ground_truth_root: Path,
    output_root: Path,
) -> dict[str, Path]:
    subset = load_ground_truth_subset(
        scene_name=scene_name,
        subset_name=subset_name,
        ground_truth_root=ground_truth_root,
    )
    if subset.scene_name != scene_name:
        raise ValueError(
            f"Subset scene_name '{subset.scene_name}' does not match requested scene '{scene_name}'."
        )

    gt_rows = load_csv_rows(subset.gt_tracks_path)
    filtered_gt_rows = filter_ground_truth_rows(gt_rows, subset)

    predictions_by_tracker = {}
    filter_check_rows = []
    for tracker_name in tracker_names:
        tracks_path = output_root / subset.scene_name / tracker_name / "tracks.csv"
        if not tracks_path.exists():
            raise FileNotFoundError(
                f"Prediction tracks CSV not found: {tracks_path}. "
                "Run the scene pipeline first."
            )
        prediction_rows = load_csv_rows(tracks_path)
        filtered_prediction_rows = filter_prediction_rows(prediction_rows, subset)
        predictions_by_tracker[tracker_name] = filtered_prediction_rows
        filter_check_rows.append(
            build_filter_check_row(
                subset=subset,
                tracker_name=tracker_name,
                gt_rows=filtered_gt_rows,
                prediction_rows=filtered_prediction_rows,
            )
        )

    workspace_root = (
        output_root
        / "experiments"
        / "gt"
        / "workspace"
        / subset.scene_name
        / subset.subset_name
    )
    workspace_paths = prepare_trackeval_workspace(
        subset=subset,
        gt_rows=filtered_gt_rows,
        predictions_by_tracker=predictions_by_tracker,
        workspace_root=workspace_root,
    )
    gt_summary_rows = evaluate_with_trackeval(
        subset=subset,
        tracker_names=tracker_names,
        filtered_gt_rows=filtered_gt_rows,
        workspace_paths=workspace_paths,
        trackeval_root=trackeval_root,
    )

    gt_root = output_root / "experiments" / "gt"
    return write_gt_outputs(
        gt_rows=gt_summary_rows,
        filter_check_rows=filter_check_rows,
        gt_root=gt_root,
    )


def _filter_rows_by_subset(
    rows: list[dict[str, str]],
    subset: GroundTruthSubset,
    require_bbox: bool,
) -> list[dict[str, str]]:
    allowed_classes = set(subset.classes)
    filtered: list[dict[str, str]] = []
    for row in rows:
        frame_idx = int(row["frame_idx"])
        class_name = str(row["class_name"])
        if frame_idx < subset.frame_start or frame_idx > subset.frame_end:
            continue
        if class_name not in allowed_classes:
            continue
        if require_bbox:
            for key in ("x1", "y1", "x2", "y2"):
                if key not in row or row[key] == "":
                    raise ValueError(f"Missing bbox field '{key}' in row: {row}")
        filtered.append(row)
    return filtered


def _convert_gt_rows_to_mot_lines(
    rows: list[dict[str, str]],
    frame_start: int,
) -> list[str]:
    lines = []
    for row in rows:
        frame_idx = int(row["frame_idx"]) - frame_start + 1
        left, top, width, height = _bbox_to_ltwh(row)
        lines.append(
            f"{frame_idx},{int(row['track_id'])},{left:.3f},{top:.3f},{width:.3f},{height:.3f},1,1,1"
        )
    return lines


def _convert_prediction_rows_to_mot_lines(
    rows: list[dict[str, str]],
    frame_start: int,
) -> list[str]:
    lines = []
    for row in rows:
        frame_idx = int(row["frame_idx"]) - frame_start + 1
        left, top, width, height = _bbox_to_ltwh(row)
        confidence = float(row.get("confidence", 1.0))
        lines.append(
            f"{frame_idx},{int(row['track_id'])},{left:.3f},{top:.3f},{width:.3f},{height:.3f},{confidence:.6f}"
        )
    return lines


def _bbox_to_ltwh(row: dict[str, str]) -> tuple[float, float, float, float]:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _probe_video_info(video_path: Path) -> tuple[int, int, int]:
    try:
        import cv2
    except ImportError:
        return 30, 1920, 1080

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 30, 1920, 1080
    try:
        fps = int(round(capture.get(cv2.CAP_PROP_FPS) or 30))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        return fps or 30, width or 1920, height or 1080
    finally:
        capture.release()


def _write_seqinfo(
    path: Path,
    seq_name: str,
    frame_count: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    lines = [
        "[Sequence]",
        f"name={seq_name}",
        "imDir=img1",
        f"frameRate={fps}",
        f"seqLength={frame_count}",
        f"imWidth={width}",
        f"imHeight={height}",
        "imExt=.jpg",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_seqmap(path: Path, seq_name: str) -> None:
    path.write_text(f"name\n{seq_name}\n", encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _import_trackeval(trackeval_root: Path):
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    candidate = trackeval_root
    if (candidate / "trackeval" / "__init__.py").exists():
        import_root = candidate
    elif candidate.name == "trackeval" and (candidate / "__init__.py").exists():
        import_root = candidate.parent
    else:
        raise FileNotFoundError(
            "TrackEval root must contain a 'trackeval' package directory."
        )

    import_root_str = str(import_root.resolve())
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
    importlib.invalidate_caches()
    try:
        return importlib.import_module("trackeval")
    except ImportError as exc:
        raise RuntimeError(f"Could not import TrackEval from {trackeval_root}") from exc


def _format_trackeval_percentage(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        scalar = sum(float(item) for item in value) / len(value) if value else 0.0
        return round(100.0 * scalar, 3)
    if hasattr(value, "tolist"):
        values = value.tolist()
        if isinstance(values, list):
            scalar = sum(float(item) for item in values) / len(values) if values else 0.0
            return round(100.0 * scalar, 3)
    return round(100.0 * float(value), 3)


def _merge_existing_rows(
    path: Path,
    new_rows: list[dict[str, object]],
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    existing_rows: list[dict[str, object]] = []
    if path.exists():
        existing_rows = [
            {key: value for key, value in row.items()}
            for row in load_csv_rows(path)
        ]

    replacement_keys = {
        tuple(str(row[field]) for field in key_fields)
        for row in new_rows
    }
    merged = [
        row
        for row in existing_rows
        if tuple(str(row[field]) for field in key_fields) not in replacement_keys
    ]
    merged.extend(new_rows)
    merged.sort(key=lambda row: tuple(str(row[field]) for field in key_fields))
    return merged


def _frame_min(rows: list[dict[str, str]]) -> int:
    if not rows:
        return -1
    return min(int(row["frame_idx"]) for row in rows)


def _frame_max(rows: list[dict[str, str]]) -> int:
    if not rows:
        return -1
    return max(int(row["frame_idx"]) for row in rows)


def _normalize_optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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
