#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isp_synth.calibration import DomainCalibration  # noqa: E402
from isp_synth.io import (  # noqa: E402
    append_jsonl,
    array_to_image,
    collect_images,
    copy_label,
    find_matching_label,
    image_to_array,
    save_image,
    stable_seed,
)


def process_one(
    calibration: DomainCalibration,
    image_path: Path,
    source_root: Path,
    labels_folder: Optional[Path],
    output_images: Path,
    output_labels: Optional[Path],
    seed: int,
    quality: int,
    overwrite: bool,
) -> Tuple[bool, Optional[Dict[str, object]]]:
    relative = image_path.relative_to(source_root)
    output_path = output_images / relative
    if output_path.exists() and not overwrite:
        return False, None

    local_seed = stable_seed(seed, relative.as_posix(), 0)
    calibrated = calibration.apply(
        image_to_array(image_path),
        rng=np.random.default_rng(local_seed),
    )
    save_image(array_to_image(calibrated), output_path, quality=quality)

    label_source = find_matching_label(image_path, labels_folder)
    output_label = None
    if label_source is not None and output_labels is not None:
        output_label = output_labels / relative.with_suffix(".txt")
        copy_label(label_source, output_label)

    record = {
        "input": str(image_path.resolve()),
        "output": str(output_path.resolve()),
        "label": (
            None if output_label is None else str(output_label.resolve())
        ),
        "seed": local_seed,
        "method": "direct_rgb_calibration_without_isp",
    }
    return True, record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply target-domain RGB calibration directly to normal-light "
            "images without running the ISP pipeline."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-glob", default="*")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not args.calibration.exists():
        raise FileNotFoundError(
            "Calibration file not found: {}".format(args.calibration)
        )

    images = [
        path
        for path in collect_images(args.images)
        if fnmatch(path.name, args.image_glob)
    ]
    if not images:
        raise FileNotFoundError(
            "No images matched {!r} in {}".format(
                args.image_glob, args.images
            )
        )

    args.output_images.mkdir(parents=True, exist_ok=True)
    if args.output_labels is not None:
        args.output_labels.mkdir(parents=True, exist_ok=True)
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else args.output_images / "direct_calibration_metadata.jsonl"
    )
    if metadata_path.exists() and args.overwrite:
        metadata_path.unlink()

    calibration = DomainCalibration.from_json(args.calibration)
    generated = 0
    skipped = 0

    def run(image_path: Path):
        return process_one(
            calibration=calibration,
            image_path=image_path,
            source_root=args.images,
            labels_folder=args.labels,
            output_images=args.output_images,
            output_labels=args.output_labels,
            seed=args.seed,
            quality=args.quality,
            overwrite=args.overwrite,
        )

    if args.workers == 1:
        iterator = (run(path) for path in images)
        for created, record in tqdm(
            iterator, total=len(images), desc="Direct calibration"
        ):
            if created and record is not None:
                append_jsonl(metadata_path, record)
                generated += 1
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run, path) for path in images]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Direct calibration",
            ):
                created, record = future.result()
                if created and record is not None:
                    append_jsonl(metadata_path, record)
                    generated += 1
                else:
                    skipped += 1

    summary = {
        "source_images": len(images),
        "generated": generated,
        "skipped": skipped,
        "calibration": str(args.calibration.resolve()),
        "output_images": str(args.output_images.resolve()),
        "output_labels": (
            None
            if args.output_labels is None
            else str(args.output_labels.resolve())
        ),
        "metadata": str(metadata_path.resolve()),
    }
    summary_path = args.output_images / "direct_calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
