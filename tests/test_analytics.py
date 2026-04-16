from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_analytics.analytics import AnalyticsEngine
from traffic_analytics.config import CountLineConfig, RuntimeConfig, ZoneConfig
from traffic_analytics.evaluation import (
    build_comparison_rows,
    build_quick_comparison_payload,
    build_run_summary,
    detect_suspected_id_handoffs,
    render_quick_comparison_markdown,
)
from traffic_analytics.tracker_backend import TrackedObject


class AnalyticsEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.line = CountLineConfig(name="midline", points=((10.0, 0.0), (10.0, 20.0)))
        self.zones = (
            ZoneConfig(
                name="north_entry",
                polygon=((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)),
            ),
            ZoneConfig(
                name="west_exit",
                polygon=((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)),
            ),
        )
        self.analytics = AnalyticsEngine(
            count_lines=(self.line,),
            zones=self.zones,
            movement_map={"north_entry": {"west_exit": "left"}},
        )

    def _track(self, frame_idx: int, point: tuple[float, float], track_id: int = 1) -> TrackedObject:
        point_x, point_y = point
        return TrackedObject(
            frame_idx=frame_idx,
            timestamp_sec=float(frame_idx),
            track_id=track_id,
            class_id=2,
            class_name="car",
            confidence=0.9,
            bbox=(point_x - 1.0, point_y - 2.0, point_x + 1.0, point_y),
            point=point,
        )

    def test_line_crossing_is_counted_once(self) -> None:
        self.analytics.process_tracks([self._track(0, (5.0, 10.0))], 0, 0.0)
        self.analytics.process_tracks([self._track(1, (15.0, 10.0))], 1, 1.0)
        self.analytics.process_tracks([self._track(2, (5.0, 10.0))], 2, 2.0)
        summary = self.analytics.finalize()

        self.assertEqual(summary["line_counts"]["midline"], 1)
        self.assertEqual(summary["duplicate_suppressed_events"], 1)

    def test_movement_classification_from_entry_to_exit(self) -> None:
        self.analytics.process_tracks([self._track(0, (4.0, 4.0))], 0, 0.0)
        self.analytics.process_tracks([self._track(1, (12.0, 12.0))], 1, 1.0)
        self.analytics.process_tracks([self._track(2, (24.0, 24.0))], 2, 2.0)
        summary = self.analytics.finalize()

        self.assertEqual(summary["movement_counts"]["left"], 1)
        self.assertEqual(summary["transition_counts"]["north_entry->west_exit"], 1)

    def test_unknown_movement_when_line_count_has_no_zone_pair(self) -> None:
        self.analytics.process_tracks([self._track(0, (5.0, 10.0), track_id=3)], 0, 0.0)
        self.analytics.process_tracks([self._track(1, (15.0, 10.0), track_id=3)], 1, 1.0)
        summary = self.analytics.finalize()

        self.assertEqual(summary["movement_counts"]["unknown"], 1)


class EvaluationTests(unittest.TestCase):
    def test_detect_suspected_id_handoff(self) -> None:
        per_track = {
            1: {
                "track_id": 1,
                "class_name": "car",
                "start_frame": 0,
                "end_frame": 10,
                "start_point": (10.0, 10.0),
                "end_point": (50.0, 50.0),
                "frame_count": 11,
            },
            2: {
                "track_id": 2,
                "class_name": "car",
                "start_frame": 12,
                "end_frame": 20,
                "start_point": (55.0, 52.0),
                "end_point": (80.0, 80.0),
                "frame_count": 9,
            },
        }
        self.assertEqual(
            detect_suspected_id_handoffs(per_track, max_gap_frames=5, max_distance_px=10.0),
            1,
        )

    def test_build_summary_and_comparison_rows(self) -> None:
        config = RuntimeConfig(
            project_root=PROJECT_ROOT,
            scene_path=PROJECT_ROOT / "configs" / "scenes" / "intersection_demo.yaml",
            video_path=PROJECT_ROOT / "data" / "intersection_demo.mp4",
            lidar_evidence_path=None,
            output_name="intersection_demo",
            output_dir=PROJECT_ROOT / "outputs" / "intersection_demo" / "bytetrack",
            comparison_dir=PROJECT_ROOT / "outputs" / "intersection_demo" / "comparison",
            model="yolov8n.pt",
            target_classes=("person", "car"),
            analytics_classes=("car",),
            count_lines=(),
            zones=(),
            movement_map={},
            active_area=None,
            tracker_name="bytetrack",
            tracker_config_path=PROJECT_ROOT / "configs" / "trackers" / "bytetrack.yaml",
            confidence=0.25,
            iou=0.45,
            device=None,
            short_track_threshold_frames=3,
            handoff_max_gap_frames=5,
            handoff_max_distance_px=20.0,
            trail_length=30,
            draw_boxes=True,
            draw_labels=True,
            draw_lines=True,
            draw_zones=True,
        )

        track_rows = [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "point_x": 5.0,
                "point_y": 10.0,
            },
            {
                "frame_idx": 1,
                "timestamp_sec": 0.1,
                "track_id": 1,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "x1": 2.0,
                "y1": 0.0,
                "x2": 12.0,
                "y2": 10.0,
                "point_x": 7.0,
                "point_y": 10.0,
            },
        ]
        analytics_summary = {
            "line_counts": {"midline": 1},
            "zone_entry_counts": {},
            "zone_exit_counts": {},
            "movement_counts": {"unknown": 1},
            "transition_counts": {},
            "duplicate_suppressed_events": 0,
            "unknown_track_ids": [1],
            "analytic_track_count": 1,
            "event_count": 1,
        }
        summary = build_run_summary(config, analytics_summary, track_rows, frame_count=2, fps=10.0)
        other_summary = {
            **summary,
            "tracker_name": "botsort",
            "line_counts": {"midline": 2},
            "movement_counts": {"unknown": 0},
            "class_track_counts": {"car": 1, "person": 1},
            "class_detection_counts": {"car": 2, "person": 3},
            "comparison_ready_metrics": {
                "total_line_crossings": 2,
                "unknown_movement_ratio": 0.0,
            },
        }

        rows = build_comparison_rows({"bytetrack": summary, "botsort": other_summary})
        self.assertTrue(any(row["metric"] == "line_counts" for row in rows))
        self.assertTrue(any(row["metric"] == "unknown_movement_ratio" for row in rows))
        self.assertTrue(any(row["metric"] == "class_track_counts" for row in rows))

        quick_payload = build_quick_comparison_payload(
            {"bytetrack": summary, "botsort": other_summary},
            rows,
        )
        markdown = render_quick_comparison_markdown(quick_payload)
        self.assertIn("Class Breakdown", markdown)
        self.assertIn("car", markdown)


if __name__ == "__main__":
    unittest.main()
