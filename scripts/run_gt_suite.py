from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.experiments import normalize_trackers
from traffic_analytics.gt_eval import discover_ground_truth_subsets, run_subset_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate every labeled GT subset under data/ground_truth/."
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
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene-name filter. Defaults to all discovered GT scenes.",
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

    discovered = discover_ground_truth_subsets(ground_truth_root)
    if args.scenes:
        scene_filter = {scene_name for scene_name in args.scenes}
        discovered = [
            (scene_name, subset_name)
            for scene_name, subset_name in discovered
            if scene_name in scene_filter
        ]

    if not discovered:
        raise FileNotFoundError(
            f"No GT subsets found under {ground_truth_root}."
        )

    print("Running GT evaluation suite")
    for scene_name, subset_name in discovered:
        print(f"- {scene_name}/{subset_name}")
        run_subset_evaluation(
            scene_name=scene_name,
            subset_name=subset_name,
            tracker_names=tracker_names,
            trackeval_root=trackeval_root,
            ground_truth_root=ground_truth_root,
            output_root=output_root,
        )

    print("Finished GT evaluation suite")


if __name__ == "__main__":
    main()
