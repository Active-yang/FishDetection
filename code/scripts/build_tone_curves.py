#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_curve_bank(count: int = 200, samples: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    curves = []
    gamma_count = max(4, int(np.ceil(count / 4.0)))
    gammas = np.linspace(0.82, 1.18, gamma_count)
    contrasts = np.linspace(2.0, 6.0, 4)
    for gamma in gammas:
        base = np.power(x, gamma)
        for contrast in contrasts:
            logistic = 1.0 / (1.0 + np.exp(-contrast * (base - 0.5)))
            logistic = (logistic - logistic[0]) / (logistic[-1] - logistic[0])
            blend = 0.35 + 0.45 * (
                (gamma - gammas.min()) / max(np.ptp(gammas), 1.0e-8)
            )
            curve = blend * logistic + (1.0 - blend) * base
            curve = np.maximum.accumulate(curve)
            curve[0], curve[-1] = 0.0, 1.0
            curves.append(curve.astype(np.float32))
    return np.stack(curves[:count], axis=0)


def numbers(text: str) -> List[float]:
    return [
        float(item)
        for item in re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", text
        )
    ]


def parse_dorf(path: Path, samples: int = 256) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    curves = []
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("I="):
            index += 1
            continue
        irradiance = numbers(lines[index].split("=", 1)[-1])
        index += 1
        while index < len(lines) and not lines[index].strip().startswith("B="):
            irradiance.extend(numbers(lines[index]))
            index += 1
        if index >= len(lines):
            break
        brightness = numbers(lines[index].split("=", 1)[-1])
        index += 1
        while index < len(lines) and not lines[index].strip().startswith("I="):
            values = numbers(lines[index])
            if values and len(values) >= 8:
                brightness.extend(values)
            elif brightness:
                break
            index += 1

        size = min(len(irradiance), len(brightness))
        if size < 16:
            continue
        x = np.asarray(irradiance[:size], dtype=np.float64)
        y = np.asarray(brightness[:size], dtype=np.float64)
        order = np.argsort(x)
        x, y = x[order], y[order]
        x = (x - x.min()) / max(x.max() - x.min(), 1.0e-12)
        y = (y - y.min()) / max(y.max() - y.min(), 1.0e-12)
        y = np.maximum.accumulate(y)
        grid = np.linspace(0.0, 1.0, samples)
        curve = np.interp(grid, x, y)
        curve = np.maximum.accumulate(curve)
        curve[0], curve[-1] = 0.0, 1.0
        curves.append(curve.astype(np.float32))

    if not curves:
        raise ValueError("No valid curves were parsed from {}".format(path))
    return np.stack(curves, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 200-curve tone mapping bank used by the ISP pipeline."
    )
    parser.add_argument("--dorf", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tone_curves.npy",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    curves = (
        parse_dorf(args.dorf, args.samples)
        if args.dorf is not None
        else synthetic_curve_bank(args.count, args.samples)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out), curves.astype(np.float32))
    print(
        "Saved {} curves with {} samples to {}".format(
            curves.shape[0], curves.shape[1], args.out
        )
    )


if __name__ == "__main__":
    main()
