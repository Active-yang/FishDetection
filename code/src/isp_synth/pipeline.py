from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve

from .calibration import DomainCalibration
from .config import resolve_project_path
from .curves import load_tone_curves

_EPS = 1.0e-8

class ISPSynthesizer:
    """Paper-style inverse ISP, RAW degradation, and forward ISP synthesis."""

    def __init__(self, full_cfg: Dict[str, Any], lut_size: int = 4096):
        self.full_cfg = full_cfg
        self.cfg = full_cfg["isp"]
        self.lut_size = int(lut_size)

        curve_path = resolve_project_path(full_cfg, self.cfg["tone_curves_path"])
        profile_path = resolve_project_path(
            full_cfg, self.cfg["camera_profiles_path"]
        )
        if not curve_path.exists():
            raise FileNotFoundError(
                "Tone curves not found: {}. Run scripts/build_tone_curves.py first.".format(
                    curve_path
                )
            )
        if not profile_path.exists():
            raise FileNotFoundError(
                "Camera profile file not found: {}".format(profile_path)
            )

        (
            self.tone_curves,
            self.inverse_tone_curves,
            self.tone_luts,
            self.inverse_tone_luts,
        ) = load_tone_curves(curve_path, self.lut_size)

        with profile_path.open("r", encoding="utf-8") as handle:
            profile_data = json.load(handle)
        self.rgb_to_xyz = np.asarray(
            profile_data["rgb_to_xyz"], dtype=np.float32
        ).reshape(3, 3)
        basis = np.asarray(
            profile_data.get("basis_xyz_to_camera", []), dtype=np.float32
        )
        if basis.ndim != 3 or basis.shape[1:] != (3, 3):
            basis = np.eye(3, dtype=np.float32)[None, ...]
        pool_size = int(
            self.cfg["ccm"].get(
                "pool_size", profile_data.get("ccm_pool_size", 11)
            )
        )
        self.xyz_to_camera_pool = self._build_ccm_pool(basis, pool_size)

        camera_profiles = profile_data.get("camera_profiles", [])
        self.camera_profiles = list(camera_profiles)

        self.calibration = None
        calibration_cfg = self.cfg.get("calibration", {})
        if calibration_cfg.get("enabled", False):
            calibration_path = resolve_project_path(
                full_cfg, calibration_cfg["path"]
            )
            if not calibration_path.exists():
                raise FileNotFoundError(
                    "Calibration file not found: {}".format(calibration_path)
                )
            self.calibration = DomainCalibration.from_json(calibration_path)

    @staticmethod
    def _build_ccm_pool(basis: np.ndarray, count: int) -> np.ndarray:
        basis = np.asarray(basis, dtype=np.float32)
        count = max(1, int(count))
        if basis.shape[0] >= count:
            return basis[:count]
        matrices = [matrix for matrix in basis]
        rng = np.random.default_rng(20250919)
        while len(matrices) < count:
            weights = rng.dirichlet(np.ones(basis.shape[0], dtype=np.float64))
            matrices.append(
                np.tensordot(
                    weights.astype(np.float32), basis, axes=(0, 0)
                ).astype(np.float32)
            )
        return np.stack(matrices, axis=0)

    @staticmethod
    def inverse_srgb_gamma(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        return np.where(
            values <= 0.04045,
            values / 12.92,
            np.power((values + 0.055) / 1.055, 2.4),
        ).astype(np.float32)

    @staticmethod
    def srgb_gamma(x: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
        return np.where(
            values <= 0.0031308,
            12.92 * values,
            1.055 * np.power(values, 1.0 / 2.4) - 0.055,
        ).astype(np.float32)

    @staticmethod
    def _apply_float_lut(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
        resolution = int(lut.size)
        position = np.clip(x, 0.0, 1.0) * float(resolution - 1)
        lower = np.floor(position).astype(np.int32)
        upper = np.minimum(lower + 1, resolution - 1)
        fraction = position - lower.astype(np.float32)
        y0 = np.take(lut, lower)
        y1 = np.take(lut, upper)
        return (y0 + (y1 - y0) * fraction).astype(np.float32)

    def _sample_tone_index(self, rng: np.random.Generator) -> int:
        tone_cfg = self.cfg["tone"]
        if tone_cfg.get("mode", "random") == "fixed":
            return int(
                np.clip(
                    int(tone_cfg.get("fixed_index", 0)),
                    0,
                    self.tone_curves.shape[0] - 1,
                )
            )
        return int(rng.integers(0, self.tone_curves.shape[0]))

    def _sample_ccm(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        ccm_cfg = self.cfg["ccm"]
        if not ccm_cfg.get("enabled", True):
            identity = np.eye(3, dtype=np.float32)
            return identity, identity, -1
        if ccm_cfg.get("mode", "pool") == "identity":
            identity = np.eye(3, dtype=np.float32)
            return identity, identity, -1

        if self.camera_profiles and ccm_cfg.get("mode") == "camera_profile":
            index = int(rng.integers(0, len(self.camera_profiles)))
            profile = self.camera_profiles[index]
            matrix_1 = np.asarray(
                profile.get("xyz_to_camera_1") or profile.get("color_matrix_1"),
                dtype=np.float32,
            ).reshape(3, 3)
            matrix_2 = profile.get("xyz_to_camera_2") or profile.get(
                "color_matrix_2"
            )
            matrix_2_array = (
                matrix_1
                if matrix_2 is None
                else np.asarray(matrix_2, dtype=np.float32).reshape(3, 3)
            )
            alpha = float(rng.random())
            xyz_to_camera = alpha * matrix_1 + (1.0 - alpha) * matrix_2_array
        else:
            index = int(rng.integers(0, self.xyz_to_camera_pool.shape[0]))
            xyz_to_camera = self.xyz_to_camera_pool[index]

        rgb_to_camera = np.matmul(xyz_to_camera, self.rgb_to_xyz)
        rgb_to_camera = rgb_to_camera.astype(np.float32)
        rgb_to_camera = rgb_to_camera + np.eye(3, dtype=np.float32) * 1.0e-6
        camera_to_rgb = np.linalg.inv(rgb_to_camera).astype(np.float32)

        identity_probability = float(ccm_cfg.get("identity_probability", 0.0))
        if identity_probability > 0.0 and rng.random() < identity_probability:
            identity = np.eye(3, dtype=np.float32)
            return identity, identity, -1
        return rgb_to_camera, camera_to_rgb, index

    def _sample_white_balance(
        self, rng: np.random.Generator
    ) -> np.ndarray:
        wb_cfg = self.cfg["white_balance"]
        if not wb_cfg.get("enabled", True):
            return np.ones(3, dtype=np.float32)

        red_low, red_high = map(float, wb_cfg.get("red_r_range", [0.0, 1.0]))
        blue_low, blue_high = map(
            float, wb_cfg.get("blue_r_range", [0.0, 1.0])
        )
        red_r = float(rng.uniform(red_low, red_high))
        blue_r = float(rng.uniform(blue_low, blue_high))

        gains = np.asarray(
            [
                1.0 / (1.2 + 2.0 * red_r),
                1.0,
                1.0 / (1.2 + blue_r),
            ],
            dtype=np.float32,
        )
        strength = float(wb_cfg.get("strength", 1.0))
        if strength != 1.0:
            gains = np.power(gains, strength).astype(np.float32)
        return gains

    @staticmethod
    def _apply_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        flat = image.reshape(-1, 3)
        transformed = np.matmul(flat, matrix.T)
        return transformed.reshape(height, width, 3).astype(np.float32)

    def _bayer_masks(
        self, height: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(height)[:, None]
        cols = np.arange(width)[None, :]
        red = (rows % 2 == 0) & (cols % 2 == 0)
        blue = (rows % 2 == 1) & (cols % 2 == 1)
        green = ~(red | blue)
        return red, green, blue

    def _mosaic(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        height, width, _ = image.shape
        red_mask, green_mask, blue_mask = self._bayer_masks(height, width)
        raw = np.zeros((height, width), dtype=np.float32)
        raw[red_mask] = image[..., 0][red_mask]
        raw[green_mask] = image[..., 1][green_mask]
        raw[blue_mask] = image[..., 2][blue_mask]
        return raw, (red_mask, green_mask, blue_mask)

    def _demosaic_malvar(
        self,
        raw: np.ndarray,
        masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        red_mask, green_mask, blue_mask = masks
        edge_mode = self.cfg["demosaic"].get("edge_mode", "mirror")

        green_at_red_blue = np.asarray(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [-1.0, 2.0, 4.0, 2.0, -1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ) / 8.0
        horizontal = np.asarray(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, -1.0, 0.0, -1.0, 0.0],
                [-1.0, 4.0, 5.0, 4.0, -1.0],
                [0.0, -1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ) / 8.0
        vertical = horizontal.T.copy()
        opposite = np.asarray(
            [
                [0.0, 0.0, -1.5, 0.0, 0.0],
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [-1.5, 0.0, 6.0, 0.0, -1.5],
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, -1.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ) / 8.0

        green_estimate = convolve(raw, green_at_red_blue, mode=edge_mode)
        horizontal_estimate = convolve(raw, horizontal, mode=edge_mode)
        vertical_estimate = convolve(raw, vertical, mode=edge_mode)
        opposite_estimate = convolve(raw, opposite, mode=edge_mode)

        rows = np.arange(raw.shape[0])[:, None]
        cols = np.arange(raw.shape[1])[None, :]
        green_in_red_rows = (rows % 2 == 0) & (cols % 2 == 1)
        green_in_blue_rows = (rows % 2 == 1) & (cols % 2 == 0)

        red = raw * red_mask.astype(np.float32)
        red += opposite_estimate * blue_mask.astype(np.float32)
        red += horizontal_estimate * green_in_red_rows.astype(np.float32)
        red += vertical_estimate * green_in_blue_rows.astype(np.float32)

        blue = raw * blue_mask.astype(np.float32)
        blue += opposite_estimate * red_mask.astype(np.float32)
        blue += vertical_estimate * green_in_red_rows.astype(np.float32)
        blue += horizontal_estimate * green_in_blue_rows.astype(np.float32)

        green = raw * green_mask.astype(np.float32)
        green += green_estimate * (red_mask | blue_mask).astype(np.float32)

        return np.stack([red, green, blue], axis=-1).astype(np.float32)

    def _add_noise(
        self, raw: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, float]:
        noise_cfg = self.cfg["noise"]
        if not noise_cfg.get("enabled", True):
            return raw, 0.0, 0.0

        shot_low, shot_high = map(
            float, noise_cfg.get("shot_noise_range", [0.0001, 0.012])
        )
        read_low, read_high = map(
            float, noise_cfg.get("read_noise_range", [1.0e-8, 2.0e-4])
        )
        shot = float(
            np.exp(
                rng.uniform(
                    math.log(max(shot_low, _EPS)),
                    math.log(max(shot_high, shot_low + _EPS)),
                )
            )
        )
        read = float(
            np.exp(
                rng.uniform(
                    math.log(max(read_low, _EPS)),
                    math.log(max(read_high, read_low + _EPS)),
                )
            )
        )
        variance = np.maximum(read + shot * np.maximum(raw, 0.0), _EPS)
        noisy = raw + rng.normal(0.0, 1.0, size=raw.shape).astype(np.float32) * np.sqrt(
            variance
        )
        return noisy.astype(np.float32), shot, read

    def _apply_exposure(
        self, raw: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float]:
        exposure_cfg = self.cfg["exposure"]
        if not exposure_cfg.get("enabled", True):
            return raw, 0.0
        low, high = map(float, exposure_cfg.get("stops", [0.0, 3.0]))
        stops = float(rng.uniform(low, high))
        return (raw / np.power(2.0, stops)).astype(np.float32), stops

    def synthesize(
        self,
        image: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        return_metadata: bool = True,
        stage_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng is None:
            rng = np.random.default_rng()
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected an RGB image with shape HxWx3, received {}".format(
                    image.shape
                )
            )
        source = np.clip(image, 0.0, 1.0)
        if stage_callback is not None:
            stage_callback("input_srgb", source)

        tone_index = self._sample_tone_index(rng)
        rgb_to_camera, camera_to_rgb, ccm_index = self._sample_ccm(rng)
        white_balance = self._sample_white_balance(rng)

        x = self.inverse_srgb_gamma(source)
        if stage_callback is not None:
            stage_callback("inverse_gamma", x)
        if self.cfg["tone"].get("enabled", True):
            x = self._apply_float_lut(
                x, self.inverse_tone_luts[tone_index]
            )
            if stage_callback is not None:
                stage_callback("inverse_tone", x)

        x = self._apply_matrix(x, rgb_to_camera)
        if stage_callback is not None:
            stage_callback("inverse_color_space", x)
        if self.cfg["white_balance"].get("enabled", True):
            x = x / white_balance.reshape(1, 1, 3)
            if stage_callback is not None:
                stage_callback("inverse_white_balance", x)
        x = np.clip(x, 0.0, 1.0)

        if self.cfg["bayer"].get("enabled", True):
            raw, masks = self._mosaic(x)
        else:
            raw = np.mean(x, axis=-1).astype(np.float32)
            red_mask, green_mask, blue_mask = self._bayer_masks(
                raw.shape[0], raw.shape[1]
            )
            masks = (red_mask, green_mask, blue_mask)
        if stage_callback is not None:
            stage_callback("bayer_mosaic", raw)

        order = self.cfg["noise"].get(
            "order", "noise_then_exposure"
        )
        if order == "noise_then_exposure":
            raw, shot_noise, read_noise = self._add_noise(raw, rng)
            if stage_callback is not None:
                stage_callback("noise_raw", raw)
            raw, exposure_stops = self._apply_exposure(raw, rng)
            if stage_callback is not None:
                stage_callback("exposure_raw", raw)
        else:
            raw, exposure_stops = self._apply_exposure(raw, rng)
            if stage_callback is not None:
                stage_callback("exposure_raw", raw)
            raw, shot_noise, read_noise = self._add_noise(raw, rng)
            if stage_callback is not None:
                stage_callback("noise_raw", raw)
        raw = np.clip(raw, 0.0, 1.0)

        if self.cfg["bayer"].get("enabled", True):
            x = self._demosaic_malvar(raw, masks)
        else:
            x = np.repeat(raw[..., None], 3, axis=-1)
        if stage_callback is not None:
            stage_callback("demosaic", x)

        if self.cfg["white_balance"].get("enabled", True):
            x = x * white_balance.reshape(1, 1, 3)
            if stage_callback is not None:
                stage_callback("forward_white_balance", x)
        x = self._apply_matrix(x, camera_to_rgb)
        if stage_callback is not None:
            stage_callback("forward_color_space", x)
        x = np.clip(x, 0.0, 1.0)

        if self.cfg["tone"].get("enabled", True):
            x = self._apply_float_lut(x, self.tone_luts[tone_index])
            if stage_callback is not None:
                stage_callback("forward_tone", x)
        x = self.srgb_gamma(x)
        if stage_callback is not None:
            stage_callback("srgb_output", x)
        if self.calibration is not None:
            x = self.calibration.apply(
                x, rng=rng, stage_callback=stage_callback
            )
        elif stage_callback is not None:
            stage_callback("final", x)
        if self.cfg["output"].get("clamp", True):
            x = np.clip(x, 0.0, 1.0)

        metadata = {
            "tone_index": tone_index,
            "ccm_index": ccm_index,
            "exposure_stops": exposure_stops,
            "exposure_factor": float(np.power(2.0, -exposure_stops)),
            "shot_noise": shot_noise,
            "read_noise": read_noise,
            "white_balance_gain": white_balance,
            "rgb_to_camera": rgb_to_camera,
            "camera_to_rgb": camera_to_rgb,
            "calibration_enabled": self.calibration is not None,
        }
        if not return_metadata:
            return x.astype(np.float32), {}
        return x.astype(np.float32), metadata

    def profile_name(self, index: int) -> Optional[str]:
        if index < 0 or index >= len(self.camera_profiles):
            return None
        return str(
            self.camera_profiles[index].get(
                "name", "profile_{}".format(index)
            )
        )
