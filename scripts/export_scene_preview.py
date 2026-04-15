from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.config import load_runtime_config
from traffic_analytics.io_utils import _require_cv2, ensure_dir
from traffic_analytics.visualization import render_scene_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a single preview frame with scene geometry overlays."
    )
    parser.add_argument("--scene", required=True, help="Path to the scene YAML.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to export. Defaults to the middle frame of the video.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder. Defaults to outputs/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output image path. Defaults to outputs/<scene>/preview/scene_preview_frame_<idx>.jpg",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_runtime_config(
        scene_path=args.scene,
        tracker_name="bytetrack",
        output_root=args.output_root,
    )
    cv2 = _require_cv2()

    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = args.frame_index if args.frame_index is not None else max(0, total_frames // 2)
        if total_frames > 0:
            frame_idx = max(0, min(frame_idx, total_frames - 1))

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(
                f"Could not read frame {frame_idx} from video: {config.video_path}"
            )
    finally:
        capture.release()

    preview = render_scene_preview(frame=frame, config=config, frame_idx=frame_idx)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = (
            config.output_dir.parent
            / "preview"
            / f"scene_preview_frame_{frame_idx}.jpg"
        )

    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not write preview image to {output_path}")

    print(f"Scene preview: {output_path}")


if __name__ == "__main__":
    main()
