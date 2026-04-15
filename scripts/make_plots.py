from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.plotting import create_experiment_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate experiment plots from aggregated CSV outputs."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene-name filter. Defaults to all scenes in metrics_summary.csv.",
    )
    parser.add_argument(
        "--experiments-root",
        default="outputs/experiments",
        help="Experiment output directory. Defaults to outputs/experiments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments_root = (PROJECT_ROOT / args.experiments_root).resolve()
    output_paths = create_experiment_plots(
        experiments_root=experiments_root,
        scenes=args.scenes,
    )

    print("Finished plot generation")
    for scene_name, paths in sorted(output_paths.items()):
        print(f"{scene_name}:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
