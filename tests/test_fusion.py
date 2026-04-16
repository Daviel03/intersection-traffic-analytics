from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.config import CountLineConfig, RuntimeConfig, ZoneConfig
from traffic_analytics.fusion import FusionSettings, fuse_track_rows_with_lidar, replay_analytics_from_track_rows
from traffic_analytics.lidar import LidarEvidence, load_lidar_evidence_csv, write_lidar_evidence_csv


class FusionTests(unittest.TestCase):
    def test_load_lidar_evidence_csv_reads_bbox_and_center(self) -> None:
        records = [
            LidarEvidence(
                frame_idx=10,
                object_id="cluster_1",
                center=(100.0, 200.0),
                support_score=0.9,
                bbox=(90.0, 150.0, 110.0, 200.0),
                range_m=18.5,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.csv"
            write_lidar_evidence_csv(records, path)
            loaded = load_lidar_evidence_csv(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].object_id, "cluster_1")
        self.assertEqual(loaded[0].bbox, (90.0, 150.0, 110.0, 200.0))
        self.assertEqual(loaded[0].center, (100.0, 200.0))
        self.assertEqual(loaded[0].range_m, 18.5)

    def test_fuse_track_rows_with_lidar_keeps_supported_tracks(self) -> None:
        track_rows = [
            {
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / 30.0,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.8,
                "x1": 10 + frame_idx * 5,
                "y1": 10,
                "x2": 30 + frame_idx * 5,
                "y2": 30,
                "point_x": 20 + frame_idx * 5,
                "point_y": 30,
            }
            for frame_idx in range(3)
        ] + [
            {
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / 30.0,
                "track_id": 2,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.7,
                "x1": 100,
                "y1": 10,
                "x2": 120,
                "y2": 30,
                "point_x": 110,
                "point_y": 30,
            }
            for frame_idx in range(3)
        ]
        lidar_records = [
            LidarEvidence(
                frame_idx=frame_idx,
                object_id=f"cluster_{frame_idx}",
                center=(20 + frame_idx * 5, 30.0),
                support_score=0.95,
                bbox=(10 + frame_idx * 5, 10.0, 30 + frame_idx * 5, 30.0),
                range_m=15.0,
            )
            for frame_idx in range(3)
        ]

        fused_rows, match_rows, diagnostics = fuse_track_rows_with_lidar(
            track_rows=track_rows,
            lidar_records=lidar_records,
            settings=FusionSettings(
                min_bbox_iou=0.05,
                max_center_distance_px=40.0,
                min_support_score=0.3,
                min_track_support_ratio=0.25,
                min_supported_frames=2,
                min_track_frames=3,
            ),
        )

        self.assertEqual({int(row["track_id"]) for row in fused_rows}, {1})
        self.assertEqual(diagnostics["lidar_supported_track_count"], 1)
        self.assertEqual(diagnostics["lidar_unsupported_track_count"], 1)
        self.assertEqual(diagnostics["suppressed_camera_only_tracks"], 1)
        self.assertTrue(any(int(row["track_confirmed"]) == 1 for row in match_rows))

    def test_replay_analytics_from_track_rows_reuses_existing_logic(self) -> None:
        config = RuntimeConfig(
            project_root=PROJECT_ROOT,
            scene_path=PROJECT_ROOT / "configs" / "scenes" / "intersection_demo.yaml",
            video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
            lidar_evidence_path=None,
            output_name="unit_scene",
            output_dir=PROJECT_ROOT / "outputs" / "unit_scene" / "bytetrack",
            comparison_dir=PROJECT_ROOT / "outputs" / "unit_scene" / "comparison",
            model="yolov8n.pt",
            target_classes=("car",),
            analytics_classes=("car",),
            count_lines=(CountLineConfig(name="midline", points=((25.0, 0.0), (25.0, 40.0))),),
            zones=(
                ZoneConfig(name="entry", polygon=((0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (0.0, 40.0))),
                ZoneConfig(name="exit", polygon=((30.0, 0.0), (60.0, 0.0), (60.0, 40.0), (30.0, 40.0))),
            ),
            movement_map={"entry": {"exit": "straight"}},
            active_area=None,
            tracker_name="bytetrack",
            tracker_config_path=PROJECT_ROOT / "configs" / "trackers" / "bytetrack.yaml",
            confidence=0.25,
            iou=0.45,
            device=None,
            short_track_threshold_frames=10,
            handoff_max_gap_frames=8,
            handoff_max_distance_px=80.0,
            trail_length=30,
            draw_boxes=True,
            draw_labels=True,
            draw_lines=True,
            draw_zones=True,
        )
        fused_rows = [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.8,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
                "point_x": 5,
                "point_y": 10,
                "lidar_supported": 1,
                "lidar_support_score": 0.9,
                "lidar_range_m": 20.0,
                "fused_confidence": 0.9,
            },
            {
                "frame_idx": 1,
                "timestamp_sec": 1 / 30.0,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.8,
                "x1": 15,
                "y1": 0,
                "x2": 25,
                "y2": 10,
                "point_x": 20,
                "point_y": 10,
                "lidar_supported": 1,
                "lidar_support_score": 0.9,
                "lidar_range_m": 19.0,
                "fused_confidence": 0.9,
            },
            {
                "frame_idx": 2,
                "timestamp_sec": 2 / 30.0,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.8,
                "x1": 35,
                "y1": 0,
                "x2": 45,
                "y2": 10,
                "point_x": 40,
                "point_y": 10,
                "lidar_supported": 1,
                "lidar_support_score": 0.9,
                "lidar_range_m": 18.0,
                "fused_confidence": 0.9,
            },
        ]

        summary, events = replay_analytics_from_track_rows(fused_rows, config)

        self.assertEqual(summary["line_counts"]["midline"], 1)
        self.assertEqual(summary["transition_counts"]["entry->exit"], 1)
        self.assertEqual(summary["movement_counts"]["straight"], 1)
        self.assertTrue(any(event.event_type == "movement_classification" for event in events))


if __name__ == "__main__":
    unittest.main()
