from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.external_data import prepare_external_scene_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Link or copy external camera / LiDAR data into the repo layout so the "
            "existing scene YAMLs work cleanly in Colab or other shared environments."
        )
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene alias, for example intersection_demo.",
    )
    parser.add_argument(
        "--camera-source",
        required=True,
        help="Absolute or Drive-mounted path to the camera video file.",
    )
    parser.add_argument(
        "--lidar-source",
        default=None,
        help=(
            "Optional LiDAR path. A CSV is linked to data/lidar/<scene>/evidence.csv. "
            "A directory or non-CSV file is linked under data/lidar/<scene>/raw/."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy data instead of using symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing linked or copied targets if they already exist.",
    )
    parser.add_argument(
        "--scene-copy",
        default=None,
        help=(
            "Optional output YAML path. If set, a scene copy is written with the resolved "
            "video_path and optional lidar_evidence_path."
        ),
    )
    parser.add_argument(
        "--base-scene",
        default=None,
        help="Optional base scene YAML for --scene-copy. Defaults to configs/scenes/<scene>.yaml.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output_name override when writing --scene-copy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_external_scene_data(
        scene_name=args.scene,
        camera_source=args.camera_source,
        lidar_source=args.lidar_source,
        copy=args.copy,
        force=args.force,
        scene_copy_path=args.scene_copy,
        base_scene_path=args.base_scene,
        output_name=args.output_name,
    )

    print("Prepared external data")
    for label, path in result.items():
        print(f"{label}: {path}")

    scene_yaml = result.get("scene_copy")
    if scene_yaml is not None:
        recommended_scene = scene_yaml
    else:
        recommended_scene = PROJECT_ROOT / "configs" / "scenes" / f"{args.scene}.yaml"

    print("")
    print("Recommended next commands:")
    print(
        f"  python scripts/compare_trackers.py --scene {recommended_scene}"
    )
    print(
        f"  python scripts/run_fusion_experiments.py --scenes {args.scene}"
    )


if __name__ == "__main__":
    main()
