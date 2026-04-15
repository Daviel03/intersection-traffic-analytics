from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one tracker on one scene YAML.")
    parser.add_argument("--scene", required=True, help="Path to the scene YAML.")
    parser.add_argument(
        "--tracker",
        required=True,
        choices=("bytetrack", "botsort"),
        help="Tracker backend to use.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder. Defaults to outputs/.",
    )
    parser.add_argument(
        "--save-trails",
        action="store_true",
        help="Draw short movement trails in the annotated video.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        scene_path=args.scene,
        tracker_name=args.tracker,
        output_root=args.output_root,
        save_trails=args.save_trails,
    )
    print(f"Finished {result.tracker_name}")
    print(f"Annotated video: {result.annotated_video_path}")
    print(f"Tracks CSV: {result.tracks_path}")
    print(f"Events CSV: {result.events_path}")
    print(f"Summary JSON: {result.summary_path}")


if __name__ == "__main__":
    main()
