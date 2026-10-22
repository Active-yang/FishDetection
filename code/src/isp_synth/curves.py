from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("A tone curve must contain at least two samples.")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values -= values.min()
    denominator = float(values.max())
    if denominator <= 0.0:
        return np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    values /= denominator
    values = np.maximum.accumulate(values)
    values[0], values[-1] = 0.0, 1.0
    return values.astype(np.float32)


def inverse_curve(curve: np.ndarray) -> np.ndarray:
    normalized = normalize_curve(curve)
    x = np.linspace(0.0, 1.0, normalized.size, dtype=np.float64)
    y = np.linspace(0.0, 1.0, normalized.size, dtype=np.float64)
    unique_y, first_indices = np.unique(normalized, return_index=True)
    unique_x = x[first_indices]
    if unique_y[0] > 0.0:
        unique_y = np.insert(unique_y, 0, 0.0)
        unique_x = np.insert(unique_x, 0, 0.0)
    if unique_y[-1] < 1.0:
        unique_y = np.append(unique_y, 1.0)
        unique_x = np.append(unique_x, 1.0)
    return np.interp(y, unique_y, unique_x).astype(np.float32)


def resample_curve(curve: np.ndarray, size: int = 4096) -> np.ndarray:
    normalized = normalize_curve(curve)
    source = np.linspace(0.0, 1.0, normalized.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    return np.interp(target, source, normalized).astype(np.float32)


def load_tone_curves(
    path: Union[str, Path], lut_size: int = 4096
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = np.load(str(path))
    if curves.ndim == 1:
        curves = curves[None, :]
    normalized = np.stack([normalize_curve(curve) for curve in curves], axis=0)
    inverses = np.stack([inverse_curve(curve) for curve in normalized], axis=0)
    forward_luts = np.stack([resample_curve(curve, lut_size) for curve in normalized])
    inverse_luts = np.stack([resample_curve(curve, lut_size) for curve in inverses])
    return normalized, inverses, forward_luts, inverse_luts
