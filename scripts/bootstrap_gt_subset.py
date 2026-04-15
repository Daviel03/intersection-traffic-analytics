from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.experiments import normalize_trackers
from traffic_analytics.gt_bootstrap import bootstrap_consensus_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap a review-ready GT subset from tracker consensus."
    )
    parser.add_argument("--scene", required=True, help="Scene name, for example intersection_demo.")
    parser.add_argument("--subset", required=True, help="Output subset name under data/ground_truth/<scene>/.")
    parser.add_argument("--frame-start", required=True, type=int, help="Inclusive start frame.")
    parser.add_argument("--frame-end", required=True, type=int, help="Inclusive end frame.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=("car",),
        help="Class names to include. Defaults to car.",
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Tracker names to use for consensus. Defaults to bytetrack botsort.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for cross-tracker matching. Defaults to 0.5.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional subset description override for subset.yaml.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root folder for tracker outputs. Defaults to outputs/.",
    )
    parser.add_argument(
        "--ground-truth-root",
        default="data/ground_truth",
        help="Root folder for GT subsets. Defaults to data/ground_truth.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tracker_names = normalize_trackers(args.trackers)
    result = bootstrap_consensus_subset(
        scene_name=args.scene,
        subset_name=args.subset,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        classes=tuple(args.classes),
        tracker_names=tracker_names,
        ground_truth_root=(PROJECT_ROOT / args.ground_truth_root).resolve(),
        output_root=(PROJECT_ROOT / args.output_root).resolve(),
        iou_threshold=float(args.iou_threshold),
        description=args.description,
    )

    print("Finished GT subset bootstrap")
    print(f"scene_name: {result.scene_name}")
    print(f"subset_name: {result.subset_name}")
    print(f"frame_range: {result.frame_start}-{result.frame_end}")
    print(f"classes: {', '.join(result.classes)}")
    print(f"gt_row_count: {result.gt_row_count}")
    print(f"gt_track_count: {result.gt_track_count}")
    print(f"subset_yaml: {result.subset_yaml_path}")
    print(f"gt_tracks_csv: {result.gt_tracks_path}")


if __name__ == "__main__":
    main()
