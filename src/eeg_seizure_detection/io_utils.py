"""Filesystem, JSON and cache-path utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import joblib
import numpy as np

from .config import CACHE_SCHEMA_VERSION, ExperimentConfig

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    return obj

def config_to_hash(cfg: ExperimentConfig) -> str:
    """Hash current values so mutable configurations cannot reuse stale caches."""
    cfg_dict = make_json_safe(asdict(cfg))
    payload = {"cache_schema_version": CACHE_SCHEMA_VERSION, "config": cfg_dict}
    return joblib.hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(data), f, ensure_ascii=False, indent=2)

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_patient_cache_path(cfg: ExperimentConfig, patient_id: str) -> Path:
    cfg_hash = config_to_hash(cfg)[:12]
    ensure_dir(cfg.cache_dir)
    return cfg.cache_dir / f'{patient_id}_features_{cfg_hash}.joblib'

def get_export_paths(cfg: ExperimentConfig) -> Dict[str, Path]:
    export_dir = ensure_dir(cfg.export_dir)
    return {'model': export_dir / cfg.export.model_filename, 'metadata': export_dir / cfg.export.metadata_filename, 'service_script': export_dir / cfg.export.service_script_filename, 'client_script': export_dir / cfg.export.client_script_filename}
