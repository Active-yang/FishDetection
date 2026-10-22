#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all tone curves in a curve bank."
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=PROJECT_ROOT / "assets" / "tone_curves.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_curves(path: Path) -> np.ndarray:
    curves = np.load(str(path))
    if curves.ndim == 1:
        curves = curves[None, :]
    if curves.ndim != 2 or curves.shape[1] < 2:
        raise ValueError(
            "Expected a curve bank with shape (num_curves, num_samples), "
            "received {}".format(curves.shape)
        )
    return np.asarray(curves, dtype=np.float64)


def main() -> None:
    args = parse_args()
    curves = load_curves(args.curves)
    count, samples = curves.shape
    x = np.linspace(0.0, 1.0, samples)
    index = np.arange(count)
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=max(count - 1, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.55, 1.0],
        left=0.055,
        right=0.94,
        bottom=0.075,
        top=0.92,
        hspace=0.30,
        wspace=0.18,
    )

    overview = fig.add_subplot(grid[0, :])
    for curve_index, curve in enumerate(curves):
        overview.plot(
            x,
            curve,
            color=cmap(norm(curve_index)),
            linewidth=0.9,
            alpha=0.38,
        )
    q05, q25, median, q75, q95 = np.quantile(
        curves, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0
    )
    overview.fill_between(
        x, q05, q95, color="#4c566a", alpha=0.10, label="5%-95%"
    )
    overview.fill_between(
        x, q25, q75, color="#4c566a", alpha=0.18, label="25%-75%"
    )
    overview.plot(
        x, median, color="#111111", linewidth=2.2, label="Median"
    )
    overview.plot(
        x,
        x,
        color="#d94841",
        linestyle="--",
        linewidth=1.3,
        label="Identity",
    )
    overview.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Normalized input",
        ylabel="Normalized output",
        title="Tone curve bank: all {} curves".format(count),
    )
    overview.grid(alpha=0.18)
    overview.legend(loc="lower right", frameon=False)

    block_size = int(np.ceil(count / 4))
    for block in range(4):
        start = block * block_size
        stop = min(start + block_size, count)
        axis = fig.add_subplot(grid[1, block])
        for curve_index in range(start, stop):
            axis.plot(
                x,
                curves[curve_index],
                color=cmap(norm(curve_index)),
                linewidth=1.0,
                alpha=0.58,
            )
        axis.plot(
            x,
            median,
            color="#111111",
            linewidth=1.7,
            alpha=0.9,
            label="Global median",
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="Input",
            title="Indices {}-{}".format(start, stop - 1),
        )
        if block == 0:
            axis.set_ylabel("Output")
        axis.grid(alpha=0.18)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(index)
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=fig.axes,
        location="right",
        fraction=0.025,
        pad=0.015,
        aspect=35,
    )
    colorbar.set_label("Curve index")

    outputs = []
    for extension in ("png", "svg", "pdf"):
        output_path = args.output_dir / "tone_curves_200.{}".format(extension)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)

    monotonic = np.all(np.diff(curves, axis=1) >= -1.0e-7)
    print(
        "Loaded {} curves with {} samples each; monotonic={}".format(
            count, samples, monotonic
        )
    )
    for output_path in outputs:
        print("Saved {}".format(output_path))


if __name__ == "__main__":
    main()
