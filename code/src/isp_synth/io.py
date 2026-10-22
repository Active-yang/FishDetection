from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def collect_images(folder: Union[str, Path], recursive: bool = True) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError("Image folder does not exist: {}".format(root))
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_to_array(path: Union[str, Path]) -> np.ndarray:
    with Image.open(str(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def array_to_image(array: np.ndarray) -> Image.Image:
    values = np.clip(np.round(np.asarray(array, dtype=np.float32) * 255.0), 0.0, 255.0)
    return Image.fromarray(values.astype(np.uint8), mode="RGB")


def save_image(image: Image.Image, path: Union[str, Path], quality: int = 95) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(str(output), quality=int(quality), subsampling=0)
    elif suffix == ".webp":
        image.save(str(output), quality=int(quality))
    else:
        image.save(str(output))


def stable_seed(base_seed: int, key: str, variant: int = 0) -> int:
    digest = hashlib.sha1("{}|{}".format(key, variant).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2 ** 31 - 1)


def find_matching_label(
    image_path: Path, labels_folder: Optional[Union[str, Path]]
) -> Optional[Path]:
    if labels_folder is None:
        sibling = image_path.with_suffix(".txt")
        return sibling if sibling.exists() else None
    candidate = Path(labels_folder) / (image_path.stem + ".txt")
    return candidate if candidate.exists() else None


def copy_label(
    label_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(label_path), str(destination))


def metadata_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: metadata_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_to_jsonable(item) for item in value]
    return value


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata_to_jsonable(record), ensure_ascii=False) + "\n")


def load_jsonl_prefix(path: Union[str, Path], count: int = 3) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= count:
                break
            if line.strip():
                records.append(json.loads(line))
    return records
