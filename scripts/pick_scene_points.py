from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.config import load_runtime_config
from traffic_analytics.io_utils import _require_cv2, ensure_dir
from traffic_analytics.visualization import render_scene_preview

WINDOW_NAME = "Scene Point Picker"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open one scene frame and click points to generate YAML-ready coordinates."
    )
    parser.add_argument("--scene", required=True, help="Path to the scene YAML.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to open. Defaults to the middle frame of the video.",
    )
    parser.add_argument(
        "--kind",
        choices=("active-area", "zone", "line"),
        default="zone",
        help="What kind of geometry you are picking.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional name for zone or line snippets.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output folder. Defaults to outputs/.",
    )
    parser.add_argument(
        "--raw-frame",
        action="store_true",
        help="Show the raw frame without current geometry overlays.",
    )
    parser.add_argument(
        "--snippet-output",
        default=None,
        help="Optional path to save the YAML snippet when you press Enter.",
    )
    return parser


class PointPicker:
    def __init__(
        self,
        base_frame,
        scene_name: str,
        frame_idx: int,
        kind: str,
        shape_name: str | None,
        default_save_path: Path,
        snippet_output_path: Path | None,
    ) -> None:
        self.cv2 = _require_cv2()
        self.base_frame = base_frame
        self.scene_name = scene_name
        self.frame_idx = frame_idx
        self.kind = kind
        self.shape_name = shape_name
        self.default_save_path = default_save_path
        self.snippet_output_path = snippet_output_path
        self.points: list[tuple[int, int]] = []
        self.cursor: tuple[int, int] | None = None
        self.message = "Left click: add point | Right click: undo | c: clear | Enter: print YAML | s: save image | q: quit"

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        self.cursor = (int(x), int(y))
        if event == self.cv2.EVENT_LBUTTONDOWN:
            if self.kind == "line" and len(self.points) >= 2:
                self.message = "Line mode only takes two points. Right click to undo or press c to clear."
                return
            point = (int(x), int(y))
            self.points.append(point)
            self.message = f"Added point {len(self.points)}: [{point[0]}, {point[1]}]"
            print(self.message)
        elif event == self.cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                self.message = f"Removed point: [{removed[0]}, {removed[1]}]"
                print(self.message)

    def render(self):
        frame = self.base_frame.copy()
        cv2 = self.cv2

        if self.cursor is not None:
            x, y = self.cursor
            cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), (70, 70, 70), 1)
            cv2.line(frame, (0, y), (frame.shape[1] - 1, y), (70, 70, 70), 1)
            cv2.putText(
                frame,
                f"({x}, {y})",
                (min(x + 12, frame.shape[1] - 160), max(20, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.points:
            points = np.array(self.points, dtype=np.int32)
            for index, (x, y) in enumerate(self.points, start=1):
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(index),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(self.points) >= 2:
                if self.kind == "line":
                    cv2.line(
                        frame,
                        tuple(points[0]),
                        tuple(points[1]),
                        (0, 0, 255),
                        2,
                    )
                else:
                    polyline = points.reshape((-1, 1, 2))
                    cv2.polylines(
                        frame,
                        [polyline],
                        len(self.points) >= 3,
                        (0, 0, 255),
                        2,
                    )

        overlay_lines = [
            f"Scene: {self.scene_name}",
            f"Frame: {self.frame_idx}",
            f"Kind: {self.kind}",
            f"Points: {len(self.points)}",
            self.message,
        ]
        y = 25
        for text in overlay_lines:
            cv2.putText(
                frame,
                text,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        return frame

    def print_snippet(self) -> None:
        snippet = format_snippet(self.kind, self.shape_name, self.points)
        if not snippet:
            self.message = "Need more points before YAML can be generated."
            print(self.message)
            return

        print("")
        print(snippet)
        print("")
        self.message = "YAML snippet printed to console."

        if self.snippet_output_path is not None:
            ensure_dir(self.snippet_output_path.parent)
            self.snippet_output_path.write_text(snippet + "\n", encoding="utf-8")
            print(f"Saved YAML snippet: {self.snippet_output_path}")
            self.message = f"YAML snippet saved to {self.snippet_output_path.name}"

    def save_image(self) -> None:
        image = self.render()
        ensure_dir(self.default_save_path.parent)
        if not self.cv2.imwrite(str(self.default_save_path), image):
            raise RuntimeError(f"Could not save picker image to {self.default_save_path}")
        print(f"Saved picker image: {self.default_save_path}")
        self.message = f"Saved image: {self.default_save_path.name}"


def format_snippet(
    kind: str,
    shape_name: str | None,
    points: list[tuple[int, int]],
) -> str:
    if kind == "line" and len(points) != 2:
        return ""
    if kind != "line" and len(points) < 3:
        return ""

    point_lines = [f"  - [{x}, {y}]" for x, y in points]

    if kind == "active-area":
        return "active_area:\n" + "\n".join(point_lines)

    if kind == "zone":
        name = shape_name or "new_zone"
        return "- name: {name}\n  polygon:\n{points}".format(
            name=name,
            points="\n".join(point_lines),
        )

    name = shape_name or "new_line"
    return "- name: {name}\n  points:\n{points}".format(
        name=name,
        points="\n".join(point_lines),
    )


def load_frame(video_path: Path, frame_idx: int | None):
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        resolved_frame_idx = frame_idx if frame_idx is not None else max(0, total_frames // 2)
        if total_frames > 0:
            resolved_frame_idx = max(0, min(resolved_frame_idx, total_frames - 1))

        capture.set(cv2.CAP_PROP_POS_FRAMES, resolved_frame_idx)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(
                f"Could not read frame {resolved_frame_idx} from video: {video_path}"
            )
        return frame, resolved_frame_idx
    finally:
        capture.release()


def main() -> None:
    args = build_parser().parse_args()
    config = load_runtime_config(
        scene_path=args.scene,
        tracker_name="bytetrack",
        output_root=args.output_root,
    )
    cv2 = _require_cv2()

    raw_frame, frame_idx = load_frame(config.video_path, args.frame_index)
    if args.raw_frame:
        base_frame = raw_frame
    else:
        base_frame = render_scene_preview(raw_frame, config, frame_idx=frame_idx)

    shape_stub = args.name or args.kind.replace("-", "_")
    default_save_path = (
        config.output_dir.parent
        / "preview"
        / f"{shape_stub}_picker_frame_{frame_idx}.jpg"
    )
    snippet_output_path = None
    if args.snippet_output:
        snippet_output_path = Path(args.snippet_output)
        if not snippet_output_path.is_absolute():
            snippet_output_path = (PROJECT_ROOT / snippet_output_path).resolve()

    picker = PointPicker(
        base_frame=base_frame,
        scene_name=config.output_name,
        frame_idx=frame_idx,
        kind=args.kind,
        shape_name=args.name,
        default_save_path=default_save_path,
        snippet_output_path=snippet_output_path,
    )

    print("Scene point picker controls:")
    print("  Left click  -> add point")
    print("  Right click -> remove last point")
    print("  c           -> clear points")
    print("  Enter       -> print YAML snippet")
    print("  s           -> save current marked-up image")
    print("  q / Esc     -> quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, picker.on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, picker.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            picker.print_snippet()
        elif key == ord("c"):
            picker.points.clear()
            picker.message = "Cleared all points."
            print(picker.message)
        elif key == ord("s"):
            picker.save_image()
        elif key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
