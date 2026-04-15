from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.experiments import (
    aggregate_experiment_rows,
    get_or_create_summary,
    normalize_scenes,
    normalize_trackers,
    write_experiment_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment metrics for one or more scenes."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names or YAML paths. Defaults to intersection_demo and intersection_behnam.",
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Tracker names to aggregate. Defaults to bytetrack botsort.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder for scene runs. Defaults to outputs/.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun requested scenes even if summary.json already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene_paths = normalize_scenes(args.scenes)
    tracker_names = normalize_trackers(args.trackers)

    summaries = []
    for scene_path in scene_paths:
        for tracker_name in tracker_names:
            summary = get_or_create_summary(
                scene_path=scene_path,
                tracker_name=tracker_name,
                output_root=args.output_root,
                force_rerun=args.force_rerun,
            )
            summaries.append(summary)

    metric_rows, transition_rows = aggregate_experiment_rows(summaries)
    experiments_root = (PROJECT_ROOT / args.output_root / "experiments").resolve()
    output_paths = write_experiment_outputs(
        metric_rows=metric_rows,
        transition_rows=transition_rows,
        experiments_root=experiments_root,
    )

    print("Finished experiment aggregation")
    for label, path in output_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
