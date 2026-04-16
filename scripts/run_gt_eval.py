from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.experiments import normalize_trackers
from traffic_analytics.gt_eval import (
    run_subset_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a labeled GT subset with TrackEval."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene name used under data/ground_truth/<scene>/ and outputs/<scene>/.",
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="Subset name under data/ground_truth/<scene>/<subset>/.",
    )
    parser.add_argument(
        "--trackeval-root",
        required=True,
        help="Path to a local TrackEval checkout.",
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Tracker names to evaluate. Defaults to bytetrack botsort.",
    )
    parser.add_argument(
        "--system-variants",
        nargs="*",
        default=None,
        help=(
            "Optional system variants to evaluate, such as camera_bytetrack or "
            "camera_lidar_bytetrack_fusion. If omitted, camera_<tracker> variants are used."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder for tracker runs. Defaults to outputs/.",
    )
    parser.add_argument(
        "--ground-truth-root",
        default="data/ground_truth",
        help="Root folder for labeled subset files. Defaults to data/ground_truth.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tracker_names = normalize_trackers(args.trackers)
    ground_truth_root = (PROJECT_ROOT / args.ground_truth_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    trackeval_root = Path(args.trackeval_root).resolve()
    output_paths = run_subset_evaluation(
        scene_name=args.scene,
        subset_name=args.subset,
        tracker_names=tracker_names,
        trackeval_root=trackeval_root,
        ground_truth_root=ground_truth_root,
        output_root=output_root,
        system_variants=tuple(args.system_variants) if args.system_variants else None,
    )

    print("Finished GT evaluation")
    for label, path in output_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
