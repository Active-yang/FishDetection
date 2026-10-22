from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["__config_path__"] = str(resolved)
    data["__project_root__"] = str(resolved.parent.parent)
    return data


def resolve_project_path(cfg: Dict[str, Any], value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg["__project_root__"]) / path


def dotted_set(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(
    cfg: Dict[str, Any], overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    result = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("Invalid override '{}'. Expected key=value.".format(item))
        key, raw_value = item.split("=", 1)
        dotted_set(result, key.strip(), yaml.safe_load(raw_value.strip()))
    return result
