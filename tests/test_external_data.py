from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.external_data import (
    default_camera_target,
    default_lidar_target,
    materialize_external_path,
    write_scene_copy,
)


class ExternalDataTests(unittest.TestCase):
    def test_default_targets_match_expected_layout(self) -> None:
        camera_source = Path("/tmp/demo_clip.mp4")
        lidar_source = Path("/tmp/evidence.csv")
        raw_lidar_source = Path("/tmp/raw_lidar")

        self.assertEqual(
            default_camera_target("intersection_demo", camera_source),
            PROJECT_ROOT / "data" / "intersection_demo.mp4",
        )
        self.assertEqual(
            default_lidar_target("intersection_demo", lidar_source),
            PROJECT_ROOT / "data" / "lidar" / "intersection_demo" / "evidence.csv",
        )
        self.assertEqual(
            default_lidar_target("intersection_demo", raw_lidar_source),
            PROJECT_ROOT / "data" / "lidar" / "intersection_demo" / "raw",
        )

    def test_materialize_external_path_copies_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source.mp4"
            target = temp_root / "linked" / "scene.mp4"
            source.write_text("video-bytes", encoding="utf-8")

            result = materialize_external_path(
                source_path=source,
                target_path=target,
                copy=True,
                force=False,
            )

            self.assertEqual(result, target.resolve())
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "video-bytes")

    def test_write_scene_copy_updates_video_and_lidar_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            base_scene = temp_root / "base.yaml"
            output_scene = temp_root / "copy.yaml"
            base_scene.write_text(
                yaml.safe_dump(
                    {
                        "video_path": "data/original.mp4",
                        "output_name": "original_scene",
                        "target_classes": ["car"],
                        "count_lines": [{"name": "entry", "points": [[0, 0], [1, 1]]}],
                        "zones": [{"name": "entry", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                        "movement_map": {"entry": {"exit": "straight"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = write_scene_copy(
                base_scene_path=base_scene,
                output_scene_path=output_scene,
                video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
                lidar_evidence_path=PROJECT_ROOT / "data" / "lidar" / "intersection_demo" / "evidence.csv",
                output_name="intersection_demo_colab",
            )

            self.assertEqual(result, output_scene.resolve())
            payload = yaml.safe_load(output_scene.read_text(encoding="utf-8"))
            self.assertEqual(payload["video_path"], "data/intersection_demo.mp4")
            self.assertEqual(
                payload["lidar_evidence_path"],
                "data/lidar/intersection_demo/evidence.csv",
            )
            self.assertEqual(payload["output_name"], "intersection_demo_colab")


if __name__ == "__main__":
    unittest.main()
