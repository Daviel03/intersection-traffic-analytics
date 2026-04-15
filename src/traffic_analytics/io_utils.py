from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Iterable[dict[str, object]], fieldnames: list[str], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(payload: dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> Any:
    cv2 = _require_cv2()
    ensure_dir(path.parent)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run `pip install -r requirements.txt` with a "
            "supported Python version such as 3.10 to 3.12."
        ) from exc
    return cv2
