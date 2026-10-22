#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isp_synth import ISPSynthesizer, apply_overrides, load_yaml  # noqa: E402
from isp_synth.io import (  # noqa: E402
    append_jsonl,
    array_to_image,
    collect_images,
    copy_label,
    find_matching_label,
    image_to_array,
    metadata_to_jsonable,
    save_image,
    stable_seed,
)


def output_name(
    source: Path, variant: int, variants: int, suffix: str
) -> str:
    if variants == 1:
        return "{}{}".format(source.stem, suffix)
    return "{}_synth{:02d}{}".format(source.stem, variant + 1, suffix)


def process_one(
    synthesizer: ISPSynthesizer,
    image_path: Path,
    source_root: Path,
    output_images: Path,
    output_labels: Optional[Path],
    labels_folder: Optional[Path],
    variant: int,
    variants: int,
    seed: int,
    suffix: str,
    quality: int,
    overwrite: bool,
) -> Tuple[bool, Optional[Dict[str, object]]]:
    relative = image_path.relative_to(source_root)
    output_suffix = suffix if suffix else image_path.suffix.lower()
    output = output_images / relative.parent / output_name(
        image_path, variant, variants, output_suffix
    )
    label_output = (
        None
        if output_labels is None
        else output_labels / relative.parent / (output.stem + ".txt")
    )

    if output.exists() and not overwrite:
        return False, None

    local_seed = stable_seed(seed, relative.as_posix(), variant)
    dark, metadata = synthesizer.synthesize(
        image_to_array(image_path),
        rng=np.random.default_rng(local_seed),
        return_metadata=True,
    )
    save_image(array_to_image(dark), output, quality=quality)

    label_source = find_matching_label(image_path, labels_folder)
    if label_source is not None and label_output is not None:
        copy_label(label_source, label_output)

    record = {
        "input": str(image_path.resolve()),
        "relative_input": relative.as_posix(),
        "output": str(output.resolve()),
        "relative_output": output.relative_to(output_images).as_posix(),
        "label": None if label_output is None else str(label_output.resolve()),
        "variant": variant + 1,
        "seed": local_seed,
        "parameters": metadata_to_jsonable(metadata),
    }
    return True, record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a labeled dark image dataset from normal-light images."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--image-glob",
        default="*",
        help="Optional filename filter for source images.",
    )
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "isp.yaml",
    )
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N source images. Zero means all images.",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to <output-images>/synthesis_metadata.jsonl.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a YAML key, for example isp.exposure.stops=[0,1].",
    )
    args = parser.parse_args()

    if args.variants < 1:
        raise ValueError("--variants must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    synthesizer = ISPSynthesizer(cfg)

    images = collect_images(args.images, recursive=not args.no_recursive)
    images = [path for path in images if fnmatch(path.name, args.image_glob)]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(
            "No source images matched {!r} in {}".format(
                args.image_glob, args.images
            )
        )

    args.output_images.mkdir(parents=True, exist_ok=True)
    if args.output_labels is not None:
        args.output_labels.mkdir(parents=True, exist_ok=True)
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else args.output_images / "synthesis_metadata.jsonl"
    )
    if metadata_path.exists() and args.overwrite:
        metadata_path.unlink()

    output_suffix = args.suffix
    if output_suffix and not output_suffix.startswith("."):
        output_suffix = "." + output_suffix

    tasks = [
        (image_path, variant)
        for image_path in images
        for variant in range(args.variants)
    ]
    generated = 0
    skipped = 0

    def run(task: Tuple[Path, int]) -> Tuple[bool, Optional[Dict[str, object]]]:
        return process_one(
            synthesizer=synthesizer,
            image_path=task[0],
            source_root=args.images,
            output_images=args.output_images,
            output_labels=args.output_labels,
            labels_folder=args.labels,
            variant=task[1],
            variants=args.variants,
            seed=seed,
            suffix=output_suffix,
            quality=args.quality,
            overwrite=args.overwrite,
        )

    if args.workers == 1:
        iterator = (run(task) for task in tasks)
        for generated_one, record in tqdm(
            iterator, total=len(tasks), desc="ISP synthesis"
        ):
            if generated_one and record is not None:
                append_jsonl(metadata_path, record)
                generated += 1
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="ISP synthesis",
            ):
                generated_one, record = future.result()
                if generated_one and record is not None:
                    append_jsonl(metadata_path, record)
                    generated += 1
                else:
                    skipped += 1

    summary = {
        "source_images": len(images),
        "variants": args.variants,
        "generated": generated,
        "skipped": skipped,
        "output_images": str(args.output_images.resolve()),
        "output_labels": (
            None
            if args.output_labels is None
            else str(args.output_labels.resolve())
        ),
        "metadata": str(metadata_path.resolve()),
    }
    summary_path = args.output_images / "synthesis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
