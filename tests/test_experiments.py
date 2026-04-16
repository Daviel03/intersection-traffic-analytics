from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.experiments import (
    METRICS_SUMMARY_FIELDS,
    TRANSITION_FIELDS,
    render_analytics_latex_table,
    render_continuity_latex_table,
    render_fusion_latex_table,
    resolve_scene_path,
    summary_to_metrics_row,
    summary_to_transition_rows,
)
from traffic_analytics.gt_eval import GT_SUMMARY_FIELDS
from traffic_analytics.io_utils import ensure_dir, write_csv
from traffic_analytics.plotting import create_experiment_plots


class ExperimentAggregationTests(unittest.TestCase):
    def test_summary_to_metrics_row_zero_fills_missing_movements(self) -> None:
        summary = {
            "output_name": "intersection_demo",
            "tracker_name": "bytetrack",
            "model": "yolov8n.pt",
            "movement_counts": {"straight": 7, "unknown": 2},
            "transition_counts": {"far_entry->foreground_exit": 7},
            "continuity_proxies": {
                "average_track_length_frames": 12.25,
                "short_track_ratio": 0.125,
                "suspected_id_handoff_count": 3,
            },
            "duplicate_suppressed_events": 4,
            "comparison_ready_metrics": {
                "total_line_crossings": 9,
                "unknown_movement_ratio": 0.222222,
            },
        }

        row = summary_to_metrics_row(summary)

        self.assertEqual(row["system_variant"], "camera_bytetrack")
        self.assertEqual(row["fusion_enabled"], 0)
        self.assertEqual(row["total_line_crossings"], 9)
        self.assertEqual(row["total_zone_transitions"], 7)
        self.assertEqual(row["left_count"], 0)
        self.assertEqual(row["right_count"], 0)
        self.assertEqual(row["straight_count"], 7)
        self.assertEqual(row["unknown_count"], 2)
        self.assertEqual(row["lidar_supported_track_count"], 0)

    def test_summary_to_transition_rows_shape(self) -> None:
        summary = {
            "output_name": "intersection_demo",
            "tracker_name": "botsort",
            "transition_counts": {
                "far_entry->foreground_exit": 19,
                "far_entry->left_exit": 1,
            },
        }

        rows = summary_to_transition_rows(summary)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["scene_name"], "intersection_demo")
        self.assertEqual(rows[0]["tracker_name"], "botsort")
        self.assertEqual(rows[0]["system_variant"], "camera_botsort")
        self.assertIn("transition_name", rows[0])
        self.assertIn("count", rows[0])

    def test_render_latex_tables_include_expected_values(self) -> None:
        metric_rows = [
            {
                "scene_name": "intersection_demo",
                "system_variant": "camera_bytetrack",
                "tracker_name": "bytetrack",
                "fusion_enabled": 0,
                "model": "yolov8n.pt",
                "total_line_crossings": 10,
                "total_zone_transitions": 8,
                "left_count": 1,
                "straight_count": 6,
                "right_count": 1,
                "unknown_count": 2,
                "unknown_movement_ratio": 0.2,
                "avg_track_length": 14.5,
                "short_track_ratio": 0.125,
                "suspected_handoff_count": 3,
                "duplicate_suppressed_events": 4,
                "lidar_supported_track_count": 0,
                "lidar_unsupported_track_count": 0,
                "fused_confirmation_events": 0,
                "suppressed_camera_only_tracks": 0,
                "average_lidar_support_ratio": 0.0,
            },
            {
                "scene_name": "intersection_demo",
                "system_variant": "camera_lidar_bytetrack_fusion",
                "tracker_name": "bytetrack",
                "fusion_enabled": 1,
                "model": "yolov8n.pt",
                "total_line_crossings": 9,
                "total_zone_transitions": 7,
                "left_count": 1,
                "straight_count": 5,
                "right_count": 1,
                "unknown_count": 1,
                "unknown_movement_ratio": 0.143,
                "avg_track_length": 18.0,
                "short_track_ratio": 0.1,
                "suspected_handoff_count": 2,
                "duplicate_suppressed_events": 3,
                "lidar_supported_track_count": 6,
                "lidar_unsupported_track_count": 2,
                "fused_confirmation_events": 6,
                "suppressed_camera_only_tracks": 2,
                "average_lidar_support_ratio": 0.625,
            }
        ]

        analytics_table = render_analytics_latex_table(metric_rows)
        continuity_table = render_continuity_latex_table(metric_rows)
        fusion_table = render_fusion_latex_table(metric_rows)

        self.assertIn("\\toprule", analytics_table)
        self.assertIn("Camera ByteTrack", analytics_table)
        self.assertIn("0.200", analytics_table)
        self.assertIn("Avg Track Length", continuity_table)
        self.assertIn("14.500", continuity_table)
        self.assertIn("Camera+LiDAR ByteTrack Fusion", fusion_table)
        self.assertIn("0.625", fusion_table)

    def test_resolve_scene_path_accepts_short_name_and_yaml_path(self) -> None:
        short_path = resolve_scene_path("intersection_demo")
        explicit_path = resolve_scene_path("configs/scenes/intersection_demo.yaml")

        self.assertTrue(short_path.name.endswith("intersection_demo.yaml"))
        self.assertEqual(short_path, explicit_path)


class PlottingTests(unittest.TestCase):
    def test_create_experiment_plots_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_root = Path(temp_dir) / "experiments"
            ensure_dir(experiments_root)
            ensure_dir(experiments_root / "gt")

            metric_rows = [
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_bytetrack",
                    "tracker_name": "bytetrack",
                    "fusion_enabled": 0,
                    "model": "yolov8n.pt",
                    "total_line_crossings": 10,
                    "total_zone_transitions": 8,
                    "left_count": 1,
                    "straight_count": 6,
                    "right_count": 1,
                    "unknown_count": 2,
                    "unknown_movement_ratio": 0.2,
                    "avg_track_length": 14.5,
                    "short_track_ratio": 0.125,
                    "suspected_handoff_count": 3,
                    "duplicate_suppressed_events": 4,
                    "lidar_supported_track_count": 0,
                    "lidar_unsupported_track_count": 0,
                    "fused_confirmation_events": 0,
                    "suppressed_camera_only_tracks": 0,
                    "average_lidar_support_ratio": 0.0,
                },
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_botsort",
                    "tracker_name": "botsort",
                    "fusion_enabled": 0,
                    "model": "yolov8n.pt",
                    "total_line_crossings": 11,
                    "total_zone_transitions": 7,
                    "left_count": 0,
                    "straight_count": 5,
                    "right_count": 1,
                    "unknown_count": 3,
                    "unknown_movement_ratio": 0.3,
                    "avg_track_length": 15.5,
                    "short_track_ratio": 0.1,
                    "suspected_handoff_count": 2,
                    "duplicate_suppressed_events": 5,
                    "lidar_supported_track_count": 0,
                    "lidar_unsupported_track_count": 0,
                    "fused_confirmation_events": 0,
                    "suppressed_camera_only_tracks": 0,
                    "average_lidar_support_ratio": 0.0,
                },
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_lidar_bytetrack_fusion",
                    "tracker_name": "bytetrack",
                    "fusion_enabled": 1,
                    "model": "yolov8n.pt",
                    "total_line_crossings": 9,
                    "total_zone_transitions": 6,
                    "left_count": 1,
                    "straight_count": 4,
                    "right_count": 1,
                    "unknown_count": 1,
                    "unknown_movement_ratio": 0.142857,
                    "avg_track_length": 16.5,
                    "short_track_ratio": 0.08,
                    "suspected_handoff_count": 1,
                    "duplicate_suppressed_events": 2,
                    "lidar_supported_track_count": 6,
                    "lidar_unsupported_track_count": 2,
                    "fused_confirmation_events": 6,
                    "suppressed_camera_only_tracks": 2,
                    "average_lidar_support_ratio": 0.65,
                },
            ]
            transition_rows = [
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_bytetrack",
                    "tracker_name": "bytetrack",
                    "transition_name": "far_entry->foreground_exit",
                    "count": 6,
                },
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_botsort",
                    "tracker_name": "botsort",
                    "transition_name": "far_entry->foreground_exit",
                    "count": 5,
                },
                {
                    "scene_name": "intersection_demo",
                    "system_variant": "camera_lidar_bytetrack_fusion",
                    "tracker_name": "bytetrack",
                    "transition_name": "far_entry->foreground_exit",
                    "count": 4,
                },
            ]
            gt_rows = [
                {
                    "scene_name": "intersection_demo",
                    "subset_name": "short_subset",
                    "system_variant": "camera_bytetrack",
                    "tracker_name": "bytetrack",
                    "frame_start": 100,
                    "frame_end": 130,
                    "num_gt_frames": 31,
                    "num_gt_tracks": 4,
                    "HOTA": 61.2,
                    "IDF1": 72.4,
                    "MOTA": 58.8,
                    "IDSW": 2,
                    "FP": 3,
                    "FN": 5,
                },
                {
                    "scene_name": "intersection_demo",
                    "subset_name": "interaction_subset",
                    "system_variant": "camera_bytetrack",
                    "tracker_name": "bytetrack",
                    "frame_start": 1646,
                    "frame_end": 1655,
                    "num_gt_frames": 10,
                    "num_gt_tracks": 3,
                    "HOTA": 71.2,
                    "IDF1": 82.4,
                    "MOTA": 68.8,
                    "IDSW": 1,
                    "FP": 4,
                    "FN": 6,
                },
                {
                    "scene_name": "intersection_demo",
                    "subset_name": "short_subset",
                    "system_variant": "camera_botsort",
                    "tracker_name": "botsort",
                    "frame_start": 100,
                    "frame_end": 130,
                    "num_gt_frames": 31,
                    "num_gt_tracks": 4,
                    "HOTA": 64.8,
                    "IDF1": 75.1,
                    "MOTA": 60.4,
                    "IDSW": 1,
                    "FP": 2,
                    "FN": 4,
                },
                {
                    "scene_name": "intersection_demo",
                    "subset_name": "interaction_subset",
                    "system_variant": "camera_botsort",
                    "tracker_name": "botsort",
                    "frame_start": 1646,
                    "frame_end": 1655,
                    "num_gt_frames": 10,
                    "num_gt_tracks": 3,
                    "HOTA": 74.8,
                    "IDF1": 85.1,
                    "MOTA": 70.4,
                    "IDSW": 0,
                    "FP": 2,
                    "FN": 5,
                },
            ]

            write_csv(metric_rows, METRICS_SUMMARY_FIELDS, experiments_root / "metrics_summary.csv")
            write_csv(
                transition_rows,
                TRANSITION_FIELDS,
                experiments_root / "transition_counts.csv",
            )
            write_csv(gt_rows, GT_SUMMARY_FIELDS, experiments_root / "gt" / "gt_eval_summary.csv")

            outputs = create_experiment_plots(experiments_root=experiments_root)
            scene_outputs = outputs["intersection_demo"]

            self.assertTrue(
                any(path.name == "movement_counts_intersection_demo.png" for path in scene_outputs)
            )
            self.assertTrue(
                any(path.name == "transitions_intersection_demo.pdf" for path in scene_outputs)
            )
            self.assertTrue(
                any(path.name == "fusion_diagnostics_intersection_demo.png" for path in scene_outputs)
            )
            self.assertTrue(
                any(path.name == "gt_metrics_intersection_demo_short_subset.png" for path in scene_outputs)
            )
            self.assertTrue(
                any(path.name == "gt_metrics_intersection_demo_interaction_subset.png" for path in scene_outputs)
            )
            for path in scene_outputs:
                self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
