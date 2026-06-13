"""
Backward-compatible data root resolver for PBJapp.

Resolution order (per logical key):
  1. Environment variable (see ``ENV_KEYS``)
  2. ``data_paths.local.json`` next to repo root (gitignored)
  3. Repo-relative default under ``repo_root()``

This module is read-only and safe to import from diagnostics. Pipeline code
should migrate to these helpers incrementally; until then, junctions at the
repo-relative paths keep legacy hardcoded paths working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_CONFIG_NAME = "data_paths.local.json"

SourceKind = Literal["env", "local_json", "repo_default", "data_root"]


@dataclass(frozen=True)
class ResolvedPath:
    key: str
    path: Path
    source: SourceKind
    env_var: Optional[str] = None


# Logical keys -> repo-relative directory name (or file for combined CSV).
DATA_ROOT_KEYS: dict[str, str] = {
    "pbjcsv": "PBJcsv",
    "standardized_pbj": "standardized_PBJ",
    "nonnursecsv": "NonNursecsv",
    "standardized_nonnurse": "standardized_NonNurse",
    "ein": "EIN",
    "provider_info": "provider_info",
    "provider_info_normalized": "provider_info_normalized",
    "provider_info_extracted": "provider_info_extracted",
    "indexed": "indexed",
    "deployments": "deployments",
    "metrics_backups": "metrics_backups",
}

ENV_KEYS: dict[str, str] = {
    "pbjcsv": "PBJ_PBJCSV",
    "standardized_pbj": "PBJ_STANDARDIZED_PBJ",
    "nonnursecsv": "PBJ_NONNURSECSV",
    "standardized_nonnurse": "PBJ_STANDARDIZED_NONNURSE",
    "ein": "PBJ_EIN",
    "provider_info": "PBJ_PROVIDER_INFO",
    "provider_info_normalized": "PBJ_PROVIDER_INFO_NORMALIZED",
    "provider_info_extracted": "PBJ_PROVIDER_INFO_EXTRACTED",
    "indexed": "PBJ_INDEXED",
    "deployments": "PBJ_DEPLOYMENTS",
    "metrics_backups": "PBJ_METRICS_BACKUPS",
}

# When set, unmapped keys resolve to <PBJ_DATA_ROOT>/<relative_name>.
GLOBAL_DATA_ROOT_ENV = "PBJ_DATA_ROOT"
GLOBAL_REPO_ROOT_ENV = "PBJ_REPO_ROOT"


def repo_root() -> Path:
    override = os.environ.get(GLOBAL_REPO_ROOT_ENV, "").strip().strip('"')
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT


def _load_local_config(root: Path) -> dict[str, Any]:
    cfg_path = root / _LOCAL_CONFIG_NAME
    if not cfg_path.is_file():
        return {}
    try:
        with cfg_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _path_from_value(value: Any, root: Path) -> Optional[Path]:
    if not value or not isinstance(value, str):
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def resolve_data_path(key: str, root: Path | None = None) -> ResolvedPath:
    """
    Resolve one logical data root.

    Raises ``KeyError`` for unknown keys (see ``DATA_ROOT_KEYS``).
    """
    if key not in DATA_ROOT_KEYS:
        raise KeyError(f"Unknown data path key: {key!r}")

    base = root or repo_root()
    rel_name = DATA_ROOT_KEYS[key]
    env_name = ENV_KEYS[key]

    env_val = os.environ.get(env_name, "").strip().strip('"')
    if env_val:
        return ResolvedPath(key=key, path=Path(env_val).expanduser().resolve(), source="env", env_var=env_name)

    cfg = _load_local_config(base)
    if key in cfg:
        p = _path_from_value(cfg[key], base)
        if p is not None:
            return ResolvedPath(key=key, path=p, source="local_json")

    data_root = os.environ.get(GLOBAL_DATA_ROOT_ENV, "").strip().strip('"')
    if not data_root and "data_root" in cfg:
        data_root = str(cfg["data_root"]).strip()
    if data_root:
        p = Path(data_root).expanduser().resolve() / rel_name
        return ResolvedPath(key=key, path=p, source="data_root", env_var=GLOBAL_DATA_ROOT_ENV)

    return ResolvedPath(key=key, path=(base / rel_name).resolve(), source="repo_default")


def resolve_all_data_paths(root: Path | None = None) -> dict[str, ResolvedPath]:
    return {key: resolve_data_path(key, root=root) for key in DATA_ROOT_KEYS}


def local_config_template() -> dict[str, str]:
    """Example ``data_paths.local.json`` (copy and edit; file is gitignored)."""
    return {
        "_comment": "Optional overrides. Paths may be absolute or repo-relative.",
        "data_root": "D:/PBJdata",
        **{k: DATA_ROOT_KEYS[k] for k in DATA_ROOT_KEYS},
    }
