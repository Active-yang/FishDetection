#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_number(token: str) -> float:
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return float(numerator) / float(denominator)
    return float(token)


def parse_matrix(value: Any) -> List[List[float]]:
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, (int, float)):
                flat.append(float(item))
            else:
                flat.extend(
                    parse_number(token)
                    for token in re.findall(
                        r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?", str(item)
                    )
                )
    else:
        flat = [
            parse_number(token)
            for token in re.findall(
                r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?",
                str(value),
            )
        ]
    if len(flat) != 9:
        raise ValueError(
            "Expected 9 matrix values, received {} from {!r}".format(
                len(flat), value
            )
        )
    return np.asarray(flat, dtype=np.float64).reshape(3, 3).tolist()


def exiftool_json(path: Path) -> Dict[str, Any]:
    command = [
        "exiftool",
        "-j",
        "-n",
        "-Make",
        "-Model",
        "-UniqueCameraModel",
        "-ColorMatrix1",
        "-ColorMatrix2",
        "-CalibrationIlluminant1",
        "-CalibrationIlluminant2",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    records = json.loads(result.stdout)
    if not records:
        raise RuntimeError("ExifTool returned no metadata for {}".format(path))
    return records[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DNG ColorMatrix1/2 into camera_profiles.json."
    )
    parser.add_argument("dng", type=Path, nargs="+")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PROJECT_ROOT / "assets" / "camera_profiles.json",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if shutil.which("exiftool") is None:
        raise SystemExit(
            "ExifTool is required. Install libimage-exiftool-perl or the Windows ExifTool executable."
        )

    with args.profiles.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    profiles = data.setdefault("camera_profiles", [])

    for dng_path in args.dng:
        metadata = exiftool_json(dng_path)
        matrix_1 = metadata.get("ColorMatrix1")
        matrix_2 = metadata.get("ColorMatrix2") or matrix_1
        if matrix_1 is None:
            print("SKIP {}: ColorMatrix1 is absent".format(dng_path))
            continue

        make = str(metadata.get("Make", "Unknown")).strip()
        model = str(
            metadata.get("UniqueCameraModel")
            or metadata.get("Model")
            or dng_path.stem
        ).strip()
        name = "{} {}".format(make, model).strip()
        profile = {
            "name": name,
            "source_file": dng_path.name,
            "matrix_direction": "XYZ_to_CameraRGB",
            "illuminant_1": metadata.get("CalibrationIlluminant1"),
            "illuminant_2": metadata.get("CalibrationIlluminant2"),
            "xyz_to_camera_1": parse_matrix(matrix_1),
            "xyz_to_camera_2": parse_matrix(matrix_2),
        }

        existing = next(
            (
                index
                for index, item in enumerate(profiles)
                if item.get("name") == name
            ),
            None,
        )
        if existing is not None and args.replace:
            profiles[existing] = profile
            print("replaced: {}".format(name))
        elif existing is not None:
            print("SKIP {}: already exists; pass --replace".format(name))
        else:
            profiles.append(profile)
            print("added: {}".format(name))

    with args.profiles.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(
        "Saved {} camera profiles to {}".format(
            len(profiles), args.profiles
        )
    )


if __name__ == "__main__":
    main()
