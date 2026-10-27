#!/usr/bin/env python3
"""Create a synthetic clip, or export CAM_FRONT frames from a local nuScenes root.

This script does not download nuScenes. Grab the mini split from
https://www.nuscenes.org/nuscenes#download and pass ``--nuscenes-root``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.teacher.synthetic import generate_synthetic_clip  # noqa: E402


def export_nuscenes_front(
    dataroot: Path,
    output: Path,
    scene_index: int = 0,
    version: str = "v1.0-mini",
    max_frames: int = 40,
) -> Path:
    """Copy one scene of CAM_FRONT images and write ``intrinsics.json``."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise SystemExit(
            "nuscenes-devkit is not installed. Use the 'nuscenes' extra in pyproject.toml."
        ) from exc
    if not dataroot.is_dir():
        raise SystemExit(f"nuScenes root does not exist: {dataroot}")
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    if scene_index < 0 or scene_index >= len(nusc.scene):
        raise SystemExit(f"scene index {scene_index} is outside 0..{len(nusc.scene) - 1}")
    scene = nusc.scene[scene_index]
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    token = scene["first_sample_token"]
    intrinsics = None
    written = 0
    while token and written < max_frames:
        sample = nusc.get("sample", token)
        cam_token = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", cam_token)
        src = Path(nusc.get_sample_data_path(cam_token))
        suffix = src.suffix or ".jpg"
        shutil.copyfile(src, frames / f"{written:06d}{suffix}")
        if intrinsics is None:
            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            matrix = calib["camera_intrinsic"]
            translation = calib["translation"]
            intrinsics = {
                "fx": matrix[0][0],
                "fy": matrix[1][1],
                "cx": matrix[0][2],
                "cy": matrix[1][2],
                "width": sd["width"],
                "height": sd["height"],
                "camera_height_m": float(translation[2]),
                "pitch_rad": 0.0,
                "nuscenes_translation": translation,
                "nuscenes_rotation": calib["rotation"],
                "note": (
                    "pitch_rad is left at 0. Decompose calibrated_sensor.rotation "
                    "before trusting the ground-plane prior on this clip."
                ),
            }
        token = sample["next"]
        written += 1
    if intrinsics is None:
        raise SystemExit(f"scene {scene['name']} has no CAM_FRONT frames")
    (output / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    (output / "prompts.json").write_text(json.dumps({"lanes": []}, indent=2), encoding="utf-8")
    print(f"exported {written} CAM_FRONT frames from {scene['name']} to {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic_clip"))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--nuscenes-root", type=Path, default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--nuscenes-version", type=str, default="v1.0-mini")
    args = parser.parse_args(argv)
    if args.nuscenes_root is not None:
        export_nuscenes_front(
            args.nuscenes_root,
            args.output,
            scene_index=args.scene_index,
            version=args.nuscenes_version,
            max_frames=args.frames,
        )
        return
    path = generate_synthetic_clip(args.output, num_frames=args.frames)
    print(f"wrote synthetic clip to {path}")


if __name__ == "__main__":
    main()
