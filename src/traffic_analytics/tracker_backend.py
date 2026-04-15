from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traffic_analytics.geometry import bottom_center

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedObject:
    frame_idx: int
    timestamp_sec: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    point: Point

    def to_csv_row(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.bbox
        point_x, point_y = self.point
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 6),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "x1": round(x1, 3),
            "y1": round(y1, 3),
            "x2": round(x2, 3),
            "y2": round(y2, 3),
            "point_x": round(point_x, 3),
            "point_y": round(point_y, 3),
        }


class UltralyticsTrackerBackend:
    def __init__(
        self,
        model_path: str,
        tracker_config_path: Path,
        target_classes: tuple[str, ...],
        confidence: float,
        iou: float,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(model_path)
        self.tracker_config_path = tracker_config_path
        self.confidence = confidence
        self.iou = iou
        self.device = device

        raw_names = self.model.names
        if isinstance(raw_names, dict):
            self.names = {int(class_id): str(name) for class_id, name in raw_names.items()}
        else:
            self.names = {class_id: str(name) for class_id, name in enumerate(raw_names)}

        name_to_id = {name: class_id for class_id, name in self.names.items()}
        missing_classes = [name for name in target_classes if name not in name_to_id]
        if missing_classes:
            raise ValueError(
                "Scene config requested class names not found in the detector labels: "
                + ", ".join(missing_classes)
            )
        self.target_class_ids = tuple(name_to_id[name] for name in target_classes)

    def track_frame(
        self,
        frame,
        frame_idx: int,
        timestamp_sec: float,
    ) -> list[TrackedObject]:
        track_kwargs = {
            "source": frame,
            "persist": True,
            "tracker": str(self.tracker_config_path),
            "conf": self.confidence,
            "iou": self.iou,
            "classes": list(self.target_class_ids),
            "verbose": False,
        }
        if self.device:
            track_kwargs["device"] = self.device

        results = self.model.track(**track_kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []

        tracked_objects: list[TrackedObject] = []
        for bbox, track_id, class_id, confidence in zip(
            xyxy,
            track_ids,
            class_ids,
            confidences,
        ):
            parsed_bbox = tuple(float(value) for value in bbox)
            tracked_objects.append(
                TrackedObject(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=self.names.get(int(class_id), str(class_id)),
                    confidence=float(confidence),
                    bbox=parsed_bbox,
                    point=bottom_center(parsed_bbox),
                )
            )
        return tracked_objects
