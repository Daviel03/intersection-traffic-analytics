from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.config import load_runtime_config
from traffic_analytics.evaluation import (
    build_comparison_payload,
    build_comparison_rows,
    build_quick_comparison_payload,
    render_quick_comparison_markdown,
)
from traffic_analytics.io_utils import ensure_dir, write_csv, write_json
from traffic_analytics.pipeline import run_pipeline

COMPARISON_FIELDS = [
    "layer",
    "metric",
    "key",
    "bytetrack",
    "botsort",
    "delta_botsort_minus_bytetrack",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run both trackers on the same scene.")
    parser.add_argument("--scene", required=True, help="Path to the scene YAML.")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder. Defaults to outputs/.",
    )
    parser.add_argument(
        "--save-trails",
        action="store_true",
        help="Draw short movement trails in both annotated videos.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    summaries = {}
    for tracker_name in ("bytetrack", "botsort"):
        result = run_pipeline(
            scene_path=args.scene,
            tracker_name=tracker_name,
            output_root=args.output_root,
            save_trails=args.save_trails,
        )
        summaries[tracker_name] = result.summary

    runtime_config = load_runtime_config(
        scene_path=args.scene,
        tracker_name="bytetrack",
        output_root=args.output_root,
    )
    ensure_dir(runtime_config.comparison_dir)

    comparison_rows = build_comparison_rows(summaries)
    comparison_payload = build_comparison_payload(summaries, comparison_rows)
    quick_payload = build_quick_comparison_payload(summaries, comparison_rows)
    quick_markdown = render_quick_comparison_markdown(quick_payload)

    comparison_csv_path = runtime_config.comparison_dir / "comparison.csv"
    comparison_json_path = runtime_config.comparison_dir / "comparison.json"
    quick_json_path = runtime_config.comparison_dir / "quick_comparison.json"
    quick_markdown_path = runtime_config.comparison_dir / "comparison.md"

    write_csv(comparison_rows, COMPARISON_FIELDS, comparison_csv_path)
    write_json(comparison_payload, comparison_json_path)
    write_json(quick_payload, quick_json_path)
    quick_markdown_path.write_text(quick_markdown, encoding="utf-8")

    print("Finished tracker comparison")
    print(f"Comparison CSV: {comparison_csv_path}")
    print(f"Comparison JSON: {comparison_json_path}")
    print(f"Quick Comparison JSON: {quick_json_path}")
    print(f"Comparison Markdown: {quick_markdown_path}")


if __name__ == "__main__":
    main()
