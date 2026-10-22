from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .io import collect_images

_EPS = 1.0e-8


def _load_small_rgb(path: Path, max_side: int = 256) -> np.ndarray:
    with Image.open(str(path)) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _load_crop_rgb(path: Path, crop_size: int = 512) -> np.ndarray:
    with Image.open(str(path)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    size = min(int(crop_size), height, width)
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return rgb[top : top + size, left : left + size]


def _sample_pixels(
    image: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    if pixels.shape[0] <= count:
        return pixels
    indices = rng.choice(pixels.shape[0], size=count, replace=False)
    return pixels[indices]


def image_color_samples(
    image_paths: Sequence[Path],
    max_images: int,
    pixels_per_image: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    if max_images > 0 and len(paths) > max_images:
        indices = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
        paths = [paths[int(index)] for index in indices]
    samples = [
        _sample_pixels(_load_small_rgb(path), pixels_per_image, rng)
        for path in paths
    ]
    if not samples:
        raise ValueError("No images were available for color calibration.")
    return np.concatenate(samples, axis=0).astype(np.float32)


def noise_statistics(
    image_paths: Sequence[Path],
    max_images: int,
    crop_size: int,
    blur_sigma: float,
    block_size: int = 16,
    flat_fraction: float = 0.2,
) -> Tuple[np.ndarray, int]:
    paths = list(image_paths)
    if max_images > 0 and len(paths) > max_images:
        indices = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
        paths = [paths[int(index)] for index in indices]
    residuals = []
    for path in paths:
        image = _load_crop_rgb(path, crop_size=crop_size)
        smooth = gaussian_filter(
            image, sigma=(blur_sigma, blur_sigma, 0.0), mode="reflect"
        )
        residual = image - smooth
        height, width, _ = residual.shape
        block = max(1, int(block_size))
        usable_height = (height // block) * block
        usable_width = (width // block) * block
        if usable_height < block or usable_width < block:
            residuals.append(residual.reshape(-1, 3))
            continue
        residual = residual[
            :usable_height, :usable_width
        ]
        image_blocks = image[
            :usable_height, :usable_width
        ].reshape(
            usable_height // block,
            block,
            usable_width // block,
            block,
            3,
        )
        residual_blocks = residual.reshape(
            usable_height // block,
            block,
            usable_width // block,
            block,
            3,
        )
        # The flatness score uses local luminance variance, which suppresses
        # scene edges and keeps low-texture sensor/JPEG noise regions.
        score = image_blocks.var(axis=(1, 3, 4))
        flat_count = max(
            1, int(np.ceil(score.size * float(flat_fraction)))
        )
        flat_indices = np.argsort(score, axis=None)[:flat_count]
        rows = flat_indices // score.shape[1]
        cols = flat_indices % score.shape[1]
        selected = residual_blocks[rows, :, cols, :, :]
        residuals.append(selected.reshape(-1, 3))
    if not residuals:
        raise ValueError("No images were available for noise calibration.")
    residual = np.concatenate(residuals, axis=0)
    return residual.std(axis=0).astype(np.float32), len(paths)


def fit_calibration(
    synthetic_images: Sequence[Path],
    target_images: Sequence[Path],
    max_images: int = 100,
    pixels_per_image: int = 4096,
    crop_size: int = 512,
    blur_sigma: float = 1.0,
    min_color_scale: float = 0.5,
    max_color_scale: float = 2.0,
    noise_scale: float = 1.0,
    minimum_extra_noise: float = 0.0,
    noise_block_size: int = 16,
    noise_flat_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    if not synthetic_images:
        raise ValueError("Synthetic image list is empty.")
    if not target_images:
        raise ValueError("Target image list is empty.")

    synthetic_pixels = image_color_samples(
        synthetic_images, max_images, pixels_per_image, seed
    )
    target_pixels = image_color_samples(
        target_images, max_images, pixels_per_image, seed + 1
    )

    source_mean = synthetic_pixels.mean(axis=0)
    target_mean = target_pixels.mean(axis=0)
    source_std = np.maximum(synthetic_pixels.std(axis=0), _EPS)
    target_std = np.maximum(target_pixels.std(axis=0), _EPS)
    color_scale = np.clip(
        target_std / source_std, min_color_scale, max_color_scale
    ).astype(np.float32)
    color_bias = (target_mean - color_scale * source_mean).astype(np.float32)

    synthetic_noise, synthetic_count = noise_statistics(
        synthetic_images,
        max_images,
        crop_size,
        blur_sigma,
        noise_block_size,
        noise_flat_fraction,
    )
    target_noise, target_count = noise_statistics(
        target_images,
        max_images,
        crop_size,
        blur_sigma,
        noise_block_size,
        noise_flat_fraction,
    )
    extra_noise = np.sqrt(
        np.maximum(target_noise ** 2 - synthetic_noise ** 2, 0.0)
    )
    extra_noise = np.maximum(
        extra_noise, max(0.0, float(minimum_extra_noise))
    )

    return {
        "version": 1,
        "method": "unpaired_moment_and_noise_calibration",
        "color": {
            "scale": color_scale.tolist(),
            "bias": color_bias.tolist(),
            "synthetic_mean": source_mean.tolist(),
            "target_mean": target_mean.tolist(),
            "synthetic_std": source_std.tolist(),
            "target_std": target_std.tolist(),
        },
        "noise": {
            "synthetic_std": synthetic_noise.tolist(),
            "target_std": target_noise.tolist(),
            "extra_std": extra_noise.tolist(),
            "blur_sigma": float(blur_sigma),
            "scale": float(noise_scale),
            "minimum_extra_std": float(minimum_extra_noise),
            "block_size": int(noise_block_size),
            "flat_fraction": float(noise_flat_fraction),
        },
        "samples": {
            "synthetic_images": synthetic_count,
            "target_images": target_count,
            "synthetic_pixels": int(synthetic_pixels.shape[0]),
            "target_pixels": int(target_pixels.shape[0]),
            "crop_size": int(crop_size),
        },
    }


class DomainCalibration:
    def __init__(self, data: Dict[str, Any]):
        color = data["color"]
        noise = data.get("noise", {})
        self.color_scale = np.asarray(color["scale"], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.color_bias = np.asarray(color["bias"], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.noise_std = np.asarray(
            noise.get("extra_std", [0.0, 0.0, 0.0]), dtype=np.float32
        ).reshape(1, 1, 3)
        self.noise_scale = float(noise.get("scale", 1.0))
        self.noise_blur = float(noise.get("blur_sigma", 1.0))

    @classmethod
    def from_json(cls, path: Path) -> "DomainCalibration":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def apply(
        self,
        image: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        stage_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> np.ndarray:
        output = np.asarray(image, dtype=np.float32) * self.color_scale
        output = output + self.color_bias
        if stage_callback is not None:
            stage_callback("calibration_color", output)
        if rng is not None and np.any(self.noise_std > 0.0):
            noise = rng.normal(
                0.0, 1.0, size=output.shape
            ).astype(np.float32)
            if self.noise_blur > 0.0:
                noise = gaussian_filter(
                    noise,
                    sigma=(self.noise_blur, self.noise_blur, 0.0),
                    mode="reflect",
                )
            output = output + noise * self.noise_std * self.noise_scale
        if stage_callback is not None:
            stage_callback("calibration_noise", output)
        final = np.clip(output, 0.0, 1.0).astype(np.float32)
        if stage_callback is not None:
            stage_callback("final", final)
        return final


def calibration_from_folder(
    synthetic_folder: Path,
    target_folder: Path,
    output_json: Path,
    synthetic_glob: str = "*",
    target_glob: str = "*",
    **kwargs: Any
) -> Dict[str, Any]:
    synthetic_images = [
        path
        for path in collect_images(synthetic_folder)
        if fnmatch(path.name, synthetic_glob)
    ]
    target_images = [
        path
        for path in collect_images(target_folder)
        if fnmatch(path.name, target_glob)
    ]
    if not synthetic_images:
        raise ValueError(
            "No synthetic images matched {!r} in {}.".format(
                synthetic_glob, synthetic_folder
            )
        )
    if not target_images:
        raise ValueError(
            "No target images matched {!r} in {}.".format(
                target_glob, target_folder
            )
        )
    calibration = fit_calibration(
        synthetic_images=synthetic_images,
        target_images=target_images,
        **kwargs
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return calibration
