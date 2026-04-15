from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.gt_bootstrap import build_consensus_rows, bootstrap_consensus_subset
from traffic_analytics.io_utils import write_csv


class GTBootstrapTests(unittest.TestCase):
    def test_build_consensus_rows_averages_matches_and_keeps_known_single_tracker_frames(self) -> None:
        predictions = {
            "bytetrack": [
                {"frame_idx": "10", "track_id": "1", "class_name": "car", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
                {"frame_idx": "11", "track_id": "1", "class_name": "car", "x1": "1", "y1": "0", "x2": "11", "y2": "10"},
                {"frame_idx": "10", "track_id": "2", "class_name": "car", "x1": "50", "y1": "50", "x2": "60", "y2": "60"},
            ],
            "botsort": [
                {"frame_idx": "10", "track_id": "7", "class_name": "car", "x1": "2", "y1": "0", "x2": "12", "y2": "10"},
                {"frame_idx": "11", "track_id": "7", "class_name": "car", "x1": "3", "y1": "0", "x2": "13", "y2": "10"},
                {"frame_idx": "12", "track_id": "7", "class_name": "car", "x1": "4", "y1": "0", "x2": "14", "y2": "10"},
            ],
        }

        rows = build_consensus_rows(
            predictions_by_tracker=predictions,
            tracker_names=("bytetrack", "botsort"),
            iou_threshold=0.5,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["frame_idx"] for row in rows], [10, 11, 12])
        self.assertEqual([row["track_id"] for row in rows], [1, 1, 1])
        self.assertEqual(rows[0]["x1"], 1.0)
        self.assertEqual(rows[2]["x1"], 4.0)

    def test_bootstrap_consensus_subset_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "outputs"
            ground_truth_root = root / "data" / "ground_truth"
            for tracker_name, rows in {
                "bytetrack": [
                    {"frame_idx": 10, "track_id": 1, "class_name": "car", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                ],
                "botsort": [
                    {"frame_idx": 10, "track_id": 7, "class_name": "car", "x1": 1, "y1": 0, "x2": 11, "y2": 10},
                ],
            }.items():
                tracks_path = output_root / "intersection_demo" / tracker_name / "tracks.csv"
                write_csv(
                    rows,
                    ["frame_idx", "track_id", "class_name", "x1", "y1", "x2", "y2"],
                    tracks_path,
                )

            scene_dir = root / "configs" / "scenes"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "intersection_demo.yaml").write_text(
                "video_path: data/intersection_demo.mp4\noutput_name: intersection_demo\n",
                encoding="utf-8",
            )

            # Patch project root expectations by writing under the temp structure and
            # temporarily re-pointing the cwd-level paths the bootstrap function resolves.
            original_project_root = PROJECT_ROOT
            self.assertTrue(original_project_root.exists())

            from traffic_analytics import gt_bootstrap as module

            original_resolve_scene_path = module.resolve_scene_path
            module.resolve_scene_path = lambda scene_name: scene_dir / f"{scene_name}.yaml"
            try:
                result = bootstrap_consensus_subset(
                    scene_name="intersection_demo",
                    subset_name="draft_subset",
                    frame_start=10,
                    frame_end=10,
                    classes=("car",),
                    tracker_names=("bytetrack", "botsort"),
                    ground_truth_root=ground_truth_root,
                    output_root=output_root,
                    iou_threshold=0.5,
                )
            finally:
                module.resolve_scene_path = original_resolve_scene_path

            self.assertEqual(result.gt_row_count, 1)
            self.assertTrue(result.subset_yaml_path.exists())
            self.assertTrue(result.gt_tracks_path.exists())


if __name__ == "__main__":
    unittest.main()
