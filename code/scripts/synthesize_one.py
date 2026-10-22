#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isp_synth import ISPSynthesizer, apply_overrides, load_yaml  # noqa: E402
from isp_synth.io import (  # noqa: E402
    array_to_image,
    image_to_array,
    metadata_to_jsonable,
    save_image,
    stable_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize one dark image.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "isp.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a YAML key, for example isp.exposure.stops=[0,1].",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    rng_seed = stable_seed(seed, args.input.name, 0)
    synthesizer = ISPSynthesizer(cfg)
    output, metadata = synthesizer.synthesize(
        image_to_array(args.input),
        rng=np.random.default_rng(rng_seed),
        return_metadata=True,
    )
    save_image(array_to_image(output), args.output)

    record = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": rng_seed,
        "parameters": metadata_to_jsonable(metadata),
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(
                metadata_to_jsonable(record), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    print(json.dumps(metadata_to_jsonable(record), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
