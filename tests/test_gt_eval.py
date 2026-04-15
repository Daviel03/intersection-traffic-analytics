from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.gt_eval import (
    GroundTruthSubset,
    build_filter_check_row,
    discover_ground_truth_subsets,
    extract_trackeval_summary_rows,
    filter_ground_truth_rows,
    filter_prediction_rows,
    load_csv_rows,
    load_ground_truth_subset,
    prepare_trackeval_workspace,
)


class GroundTruthEvalTests(unittest.TestCase):
    def test_discover_ground_truth_subsets_finds_scene_subset_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_root = Path(temp_dir)
            for scene_name, subset_name in (
                ("intersection_demo", "short_subset"),
                ("intersection_behnam", "roundabout_turn_subset"),
            ):
                subset_dir = gt_root / scene_name / subset_name
                subset_dir.mkdir(parents=True, exist_ok=True)
                (subset_dir / "subset.yaml").write_text("scene_name: demo\n", encoding="utf-8")
                (subset_dir / "gt_tracks.csv").write_text(
                    "frame_idx,track_id,class_name,x1,y1,x2,y2\n",
                    encoding="utf-8",
                )

            discovered = discover_ground_truth_subsets(gt_root)

            self.assertEqual(
                discovered,
                [
                    ("intersection_behnam", "roundabout_turn_subset"),
                    ("intersection_demo", "short_subset"),
                ],
            )

    def test_load_ground_truth_subset_reads_optional_description(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_root = Path(temp_dir)
            subset_dir = gt_root / "intersection_demo" / "short_subset"
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_payload = {
                "scene_name": "intersection_demo",
                "video_path": "data/intersection_demo.mp4",
                "frame_start": 100,
                "frame_end": 130,
                "classes": ["car", "truck"],
                "description": "Short clip with turning vehicles.",
            }
            (subset_dir / "subset.yaml").write_text(
                yaml.safe_dump(subset_payload, sort_keys=False),
                encoding="utf-8",
            )
            (subset_dir / "gt_tracks.csv").write_text(
                "frame_idx,track_id,class_name,x1,y1,x2,y2\n",
                encoding="utf-8",
            )

            subset = load_ground_truth_subset(
                scene_name="intersection_demo",
                subset_name="short_subset",
                ground_truth_root=gt_root,
            )

            self.assertEqual(subset.description, "Short clip with turning vehicles.")
            self.assertEqual(subset.classes, ("car", "truck"))

    def test_filter_rows_match_frame_range_and_class_list(self) -> None:
        subset = GroundTruthSubset(
            scene_name="intersection_demo",
            subset_name="short_subset",
            video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
            frame_start=100,
            frame_end=102,
            classes=("car",),
            gt_tracks_path=PROJECT_ROOT / "data" / "ground_truth" / "fake.csv",
        )
        gt_rows = [
            {"frame_idx": "99", "track_id": "1", "class_name": "car", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
            {"frame_idx": "100", "track_id": "1", "class_name": "car", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
            {"frame_idx": "101", "track_id": "2", "class_name": "truck", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
            {"frame_idx": "102", "track_id": "3", "class_name": "car", "x1": "0", "y1": "0", "x2": "12", "y2": "10"},
        ]
        prediction_rows = [
            {"frame_idx": "100", "track_id": "11", "class_name": "car", "confidence": "0.9", "x1": "1", "y1": "2", "x2": "11", "y2": "12"},
            {"frame_idx": "102", "track_id": "12", "class_name": "bus", "confidence": "0.9", "x1": "1", "y1": "2", "x2": "11", "y2": "12"},
            {"frame_idx": "103", "track_id": "13", "class_name": "car", "confidence": "0.9", "x1": "1", "y1": "2", "x2": "11", "y2": "12"},
        ]

        filtered_gt = filter_ground_truth_rows(gt_rows, subset)
        filtered_predictions = filter_prediction_rows(prediction_rows, subset)
        filter_row = build_filter_check_row(
            subset=subset,
            tracker_name="bytetrack",
            gt_rows=filtered_gt,
            prediction_rows=filtered_predictions,
        )

        self.assertEqual(len(filtered_gt), 2)
        self.assertEqual(len(filtered_predictions), 1)
        self.assertEqual(filter_row["frame_start"], 100)
        self.assertEqual(filter_row["frame_end"], 102)
        self.assertEqual(filter_row["classes"], "car")
        self.assertEqual(filter_row["gt_frame_min"], 100)
        self.assertEqual(filter_row["pred_frame_max"], 100)

    def test_prepare_trackeval_workspace_converts_to_mot_style(self) -> None:
        subset = GroundTruthSubset(
            scene_name="intersection_demo",
            subset_name="short_subset",
            video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
            frame_start=100,
            frame_end=101,
            classes=("car",),
            gt_tracks_path=PROJECT_ROOT / "data" / "ground_truth" / "fake.csv",
        )
        gt_rows = [
            {"frame_idx": "100", "track_id": "1", "class_name": "car", "x1": "10", "y1": "20", "x2": "30", "y2": "50"},
            {"frame_idx": "101", "track_id": "1", "class_name": "car", "x1": "12", "y1": "21", "x2": "32", "y2": "51"},
        ]
        prediction_rows = {
            "bytetrack": [
                {
                    "frame_idx": "100",
                    "track_id": "11",
                    "class_name": "car",
                    "confidence": "0.8",
                    "x1": "11",
                    "y1": "20",
                    "x2": "31",
                    "y2": "50",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = prepare_trackeval_workspace(
                subset=subset,
                gt_rows=gt_rows,
                predictions_by_tracker=prediction_rows,
                workspace_root=Path(temp_dir) / "workspace",
            )

            gt_txt = (workspace["gt_root"] / "intersection_demo_short_subset" / "gt" / "gt.txt").read_text(encoding="utf-8")
            tracker_txt = (workspace["trackers_root"] / "bytetrack" / "data" / "intersection_demo_short_subset.txt").read_text(encoding="utf-8")

            self.assertIn("1,1,10.000,20.000,20.000,30.000,1,1,1", gt_txt)
            self.assertIn("1,11,11.000,20.000,20.000,30.000,0.800000", tracker_txt)

    def test_extract_trackeval_summary_rows(self) -> None:
        subset = GroundTruthSubset(
            scene_name="intersection_demo",
            subset_name="short_subset",
            video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
            frame_start=100,
            frame_end=102,
            classes=("car",),
            gt_tracks_path=PROJECT_ROOT / "data" / "ground_truth" / "fake.csv",
        )
        filtered_gt_rows = [
            {"frame_idx": "100", "track_id": "1", "class_name": "car", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
            {"frame_idx": "101", "track_id": "2", "class_name": "car", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
        ]
        output_res = {
            "MotChallenge2DBox": {
                "bytetrack": {
                    "COMBINED_SEQ": {
                        "pedestrian": {
                            "HOTA": {"HOTA": [0.6, 0.8]},
                            "CLEAR": {"MOTA": 0.5, "IDSW": 2, "CLR_FP": 3, "CLR_FN": 4},
                            "Identity": {"IDF1": 0.7},
                        }
                    }
                },
                "botsort": {
                    "COMBINED_SEQ": {
                        "pedestrian": {
                            "HOTA": {"HOTA": [0.65, 0.75]},
                            "CLEAR": {"MOTA": 0.55, "IDSW": 1, "CLR_FP": 2, "CLR_FN": 3},
                            "Identity": {"IDF1": 0.72},
                        }
                    }
                },
            }
        }
        output_msg = {"MotChallenge2DBox": {"bytetrack": "ok", "botsort": "ok"}}

        rows = extract_trackeval_summary_rows(
            subset=subset,
            tracker_names=("bytetrack", "botsort"),
            filtered_gt_rows=filtered_gt_rows,
            output_res=output_res,
            output_msg=output_msg,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["num_gt_frames"], 2)
        self.assertEqual(rows[0]["num_gt_tracks"], 2)
        self.assertEqual(rows[0]["HOTA"], 70.0)
        self.assertEqual(rows[0]["IDF1"], 70.0)
        self.assertEqual(rows[0]["MOTA"], 50.0)
        self.assertEqual(rows[0]["IDSW"], 2)
        self.assertEqual(rows[0]["FP"], 3)
        self.assertEqual(rows[0]["FN"], 4)


if __name__ == "__main__":
    unittest.main()
