#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isp_synth.io import collect_images  # noqa: E402


def write_image_list(path: Path, images: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(image.resolve()) for image in images) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Ultralytics YOLO data YAML that combines normal-light, "
            "real-dark, and ISP-synthetic image folders without copying files."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--original-train-images",
        type=Path,
        default=PROJECT_ROOT / "mydata" / "dark" / "train" / "images",
    )
    parser.add_argument(
        "--synthetic-images",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "calibration_preview"
        / "images",
    )
    parser.add_argument(
        "--original-glob",
        default="*",
        help="Optional filename filter for original training images.",
    )
    parser.add_argument(
        "--synthetic-glob",
        default="*",
        help="Optional filename filter for synthetic training images.",
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=PROJECT_ROOT / "mydata" / "dark" / "val" / "images",
    )
    parser.add_argument(
        "--test-images",
        type=Path,
        default=PROJECT_ROOT / "mydata" / "dark" / "test" / "images",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "calibration_preview"
        / "yolo_bright_real_isp.yaml",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "calibration_preview"
        / "train_bright_real_isp.txt",
    )
    parser.add_argument("--class-name", default="fish")
    args = parser.parse_args()

    original = collect_images(args.original_train_images)
    synthetic = collect_images(args.synthetic_images)
    original = [
        path for path in original if fnmatch(path.name, args.original_glob)
    ]
    synthetic = [
        path for path in synthetic if fnmatch(path.name, args.synthetic_glob)
    ]
    if not original:
        raise FileNotFoundError(
            "No original training images found in {}".format(
                args.original_train_images
            )
        )
    if not synthetic:
        raise FileNotFoundError(
            "No synthetic images found in {}".format(args.synthetic_images)
        )

    train_images = sorted(original + synthetic)
    write_image_list(args.train_list, train_images)

    data = {
        "path": str(args.project_root.resolve()),
        "train": str(args.train_list.resolve()),
        "val": str(args.val_images.resolve()),
        "test": str(args.test_images.resolve()),
        "names": {0: args.class_name},
    }
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("Original training images: {}".format(len(original)))
    print("Synthetic training images: {}".format(len(synthetic)))
    print("Total training images: {}".format(len(train_images)))
    print("YAML: {}".format(args.output_yaml.resolve()))
    print("Train list: {}".format(args.train_list.resolve()))


if __name__ == "__main__":
    main()
