from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from traffic_analytics.config import PROJECT_ROOT
from traffic_analytics.io_utils import ensure_dir


def default_camera_target(scene_name: str, source_path: Path) -> Path:
    suffix = source_path.suffix or ".mp4"
    return PROJECT_ROOT / "data" / f"{scene_name}{suffix}"


def default_lidar_target(scene_name: str, source_path: Path) -> Path:
    scene_root = PROJECT_ROOT / "data" / "lidar" / scene_name
    if source_path.is_dir() or source_path.suffix == "":
        return scene_root / "raw"
    if source_path.suffix.lower() == ".csv":
        return scene_root / "evidence.csv"
    return scene_root / "raw" / source_path.name


def materialize_external_path(
    source_path: str | Path,
    target_path: str | Path,
    copy: bool = False,
    force: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"External data source not found: {source}")

    ensure_dir(target.parent)
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(
                f"Target already exists: {target}. Use --force to replace it."
            )
        _remove_existing(target)

    if copy:
        _copy_source(source, target)
        return target

    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        _copy_source(source, target)
    return target


def write_scene_copy(
    base_scene_path: str | Path,
    output_scene_path: str | Path,
    video_path: str | Path,
    lidar_evidence_path: str | Path | None = None,
    output_name: str | None = None,
) -> Path:
    base_path = Path(base_scene_path).resolve()
    output_path = Path(output_scene_path).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {base_path}")

    payload["video_path"] = _to_project_relative(video_path)
    if lidar_evidence_path is not None:
        payload["lidar_evidence_path"] = _to_project_relative(lidar_evidence_path)
    if output_name:
        payload["output_name"] = output_name

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path


def prepare_external_scene_data(
    scene_name: str,
    camera_source: str | Path,
    lidar_source: str | Path | None = None,
    copy: bool = False,
    force: bool = False,
    scene_copy_path: str | Path | None = None,
    base_scene_path: str | Path | None = None,
    output_name: str | None = None,
) -> dict[str, Path]:
    camera_source_path = Path(camera_source).expanduser().resolve()
    camera_target = default_camera_target(scene_name, camera_source_path)
    materialized_camera = materialize_external_path(
        source_path=camera_source_path,
        target_path=camera_target,
        copy=copy,
        force=force,
    )

    result = {
        "camera_target": materialized_camera,
    }

    lidar_target: Path | None = None
    if lidar_source is not None:
        lidar_source_path = Path(lidar_source).expanduser().resolve()
        lidar_target = default_lidar_target(scene_name, lidar_source_path)
        materialized_lidar = materialize_external_path(
            source_path=lidar_source_path,
            target_path=lidar_target,
            copy=copy,
            force=force,
        )
        result["lidar_target"] = materialized_lidar

    if scene_copy_path is not None:
        base_scene = (
            Path(base_scene_path).resolve()
            if base_scene_path is not None
            else (PROJECT_ROOT / "configs" / "scenes" / f"{scene_name}.yaml").resolve()
        )
        result["scene_copy"] = write_scene_copy(
            base_scene_path=base_scene,
            output_scene_path=scene_copy_path,
            video_path=materialized_camera,
            lidar_evidence_path=lidar_target if lidar_target and lidar_target.suffix.lower() == ".csv" else None,
            output_name=output_name,
        )

    return result


def _copy_source(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
        return
    shutil.copy2(source, target)


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    raise FileNotFoundError(f"Could not remove missing path: {path}")


def _to_project_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)
