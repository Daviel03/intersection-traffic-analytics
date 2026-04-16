from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.fusion import FusionSettings
from traffic_analytics.fusion_experiments import run_and_write_fusion_experiments
from traffic_analytics.plotting import create_experiment_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run camera-only baselines plus camera+LiDAR late-fusion experiments."
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
        help="Camera-only tracker baselines. Defaults to bytetrack botsort.",
    )
    parser.add_argument(
        "--fusion-trackers",
        nargs="*",
        default=("bytetrack",),
        help="Base trackers used for fused variants. Defaults to bytetrack.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder for scene runs. Defaults to outputs/.",
    )
    parser.add_argument(
        "--lidar-root",
        default="data/lidar",
        help="Root folder for per-scene LiDAR evidence CSV files. Defaults to data/lidar/.",
    )
    parser.add_argument(
        "--no-mock-lidar",
        action="store_true",
        help="Require precomputed LiDAR evidence instead of generating mock motion-based evidence.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun requested camera and fusion variants even if summaries already exist.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also regenerate experiment plots after writing the aggregated CSVs and tables.",
    )
    parser.add_argument("--min-bbox-iou", type=float, default=0.05)
    parser.add_argument("--max-center-distance-px", type=float, default=80.0)
    parser.add_argument("--min-support-score", type=float, default=0.3)
    parser.add_argument("--min-track-support-ratio", type=float, default=0.25)
    parser.add_argument("--min-supported-frames", type=int, default=2)
    parser.add_argument("--min-track-frames", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    lidar_root = (PROJECT_ROOT / args.lidar_root).resolve()
    settings = FusionSettings(
        min_bbox_iou=args.min_bbox_iou,
        max_center_distance_px=args.max_center_distance_px,
        min_support_score=args.min_support_score,
        min_track_support_ratio=args.min_track_support_ratio,
        min_supported_frames=args.min_supported_frames,
        min_track_frames=args.min_track_frames,
    )

    output_paths = run_and_write_fusion_experiments(
        scenes=args.scenes,
        trackers=args.trackers,
        fusion_trackers=list(args.fusion_trackers) if args.fusion_trackers else None,
        output_root=output_root,
        lidar_root=lidar_root,
        allow_mock_lidar=not args.no_mock_lidar,
        force_rerun=args.force_rerun,
        fusion_settings=settings,
    )

    print("Finished camera + LiDAR experiment aggregation")
    for label, path in output_paths.items():
        print(f"{label}: {path}")

    if args.make_plots:
        plot_outputs = create_experiment_plots(output_root / "experiments", scenes=args.scenes)
        print("Finished plot generation")
        for scene_name, paths in sorted(plot_outputs.items()):
            print(f"{scene_name}:")
            for path in paths:
                print(f"  {path}")


if __name__ == "__main__":
    main()
