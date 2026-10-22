#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isp_synth.calibration import calibration_from_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn target-domain color and noise calibration from real dark "
            "images. Detection labels are not required."
        )
    )
    parser.add_argument(
        "--source-images",
        "--synthetic-images",
        dest="source_images",
        type=Path,
        required=True,
        help=(
            "Source distribution images. Use ISP output for residual "
            "calibration, or normal-light images for direct RGB calibration."
        ),
    )
    parser.add_argument("--target-images", type=Path, required=True)
    parser.add_argument(
        "--source-glob",
        "--synthetic-glob",
        dest="source_glob",
        default="*",
        help="Optional filename filter for source images.",
    )
    parser.add_argument(
        "--target-glob",
        default="*",
        help="Optional filename filter for real dark images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration.json",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--pixels-per-image", type=int, default=4096)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--minimum-extra-noise",
        type=float,
        default=0.0,
        help=(
            "Optional noise floor per RGB channel when measured target noise "
            "does not exceed synthetic noise."
        ),
    )
    parser.add_argument("--noise-block-size", type=int, default=16)
    parser.add_argument("--noise-flat-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    calibration = calibration_from_folder(
        synthetic_folder=args.source_images,
        target_folder=args.target_images,
        output_json=args.output,
        synthetic_glob=args.source_glob,
        target_glob=args.target_glob,
        max_images=args.max_images,
        pixels_per_image=args.pixels_per_image,
        crop_size=args.crop_size,
        blur_sigma=args.blur_sigma,
        noise_scale=args.noise_scale,
        minimum_extra_noise=args.minimum_extra_noise,
        noise_block_size=args.noise_block_size,
        noise_flat_fraction=args.noise_flat_fraction,
        seed=args.seed,
    )
    print(json.dumps(calibration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
