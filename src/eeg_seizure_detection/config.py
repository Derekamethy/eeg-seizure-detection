"""Configuration dataclasses and canonical defaults for the EEG pipeline."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Sequence

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

GLOBAL_SEED = 42

@dataclass
class FeatureConfig:
    epoch_len_s: int = 2
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 50.0
    bandpass_method: str = 'butter_sos'
    butter_order: int = 4
    band_edges_hz: Tuple[int, ...] = tuple(range(0, 42, 2))
    synchrony_pairs_names: Tuple[Tuple[str, str], ...] = (('FP1-F3', 'FP2-F4'), ('F7-T7', 'F8-T8'), ('C3-P3', 'C4-P4'))
    history_epochs: int = 3
    scale_to_uV: bool = True
    channel_missing_policy: str = 'strict'

@dataclass
class EvalConfig:
    patient_ids: Tuple[str, ...] = tuple((f'chb{i:02d}' for i in range(1, 11)))
    threshold_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.1, 0.95, 50), 3))
    min_duration_epochs: int = 3
    min_acceptable_sensitivity: float = 0.7
    neg_to_pos_ratio: int = 10
    min_neg_samples: int = 3000
    default_threshold: float = 0.5
    fixed_threshold_mode: bool = True
    fixed_threshold_value: float = 0.5
    svm_c: float = 1.0
    svm_gamma: str = 'scale'
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 1
    rf_max_features: str = 'sqrt'
    rf_n_jobs: int = -1
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 1.0
    xgb_gamma: float = 0.0
    xgb_tree_method: str = 'hist'
    xgb_n_jobs: int = -1
    top_k_features: int = 50
    selector_n_estimators: int = 80
    selector_max_depth: int = 4
    random_state: int = GLOBAL_SEED

@dataclass
class ExportConfig:
    export_subdir: str = 'exported_models'
    model_filename: str = 'rf_lightweight_model.joblib'
    metadata_filename: str = 'rf_lightweight_metadata.json'
    service_script_filename: str = 'serve_model.py'
    client_script_filename: str = 'example_client.py'

@dataclass
class ExperimentConfig:
    data_root: str = 'data/chb-mit'
    cache_subdir: str = 'feature_cache_refactored'
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def channels(self) -> List[str]:
        """Cached: avoids rebuilding the list on every access (optimization #10+#17)."""
        if not hasattr(self, '_channels_cached'):
            ref_ch_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']
            self._channels_cached = list(dict.fromkeys(ref_ch_names))
        return self._channels_cached

    @property
    def cache_dir(self) -> Path:
        return Path(self.data_root) / self.cache_subdir

    @property
    def export_dir(self) -> Path:
        return Path(self.data_root) / self.export.export_subdir

CFG = ExperimentConfig()

CACHE_SCHEMA_VERSION = 'validated_annotations'


def make_final_rf_config(data_root: str | Path, patient_ids: Sequence[str] | None = None) -> ExperimentConfig:
    """Return the reported final RF configuration with a caller-owned data path."""
    cfg = copy.deepcopy(CFG)
    cfg.data_root = str(data_root)
    cfg.feature.history_epochs = 3
    cfg.feature.bandpass_method = "butter_sos"
    cfg.feature.butter_order = 4
    cfg.eval.top_k_features = 30
    cfg.eval.rf_n_estimators = 500
    cfg.eval.rf_max_depth = 12
    cfg.eval.rf_min_samples_leaf = 1
    cfg.eval.rf_max_features = "sqrt"
    cfg.eval.fixed_threshold_mode = False
    if patient_ids is not None:
        cfg.eval.patient_ids = tuple(patient_ids)
    return cfg


def load_config(path: str | Path, data_root: str | Path, patient_ids: Sequence[str] | None = None) -> ExperimentConfig:
    """Load the executable JSON configuration; reject unknown options."""
    import json
    with open(path, encoding="utf-8") as stream:
        values = json.load(stream)
    unknown = set(values) - {"feature", "eval", "export"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    cfg = ExperimentConfig(data_root=str(data_root),
                           feature=FeatureConfig(**values.get("feature", {})),
                           eval=EvalConfig(**values.get("eval", {})),
                           export=ExportConfig(**values.get("export", {})))
    if patient_ids is not None:
        cfg.eval.patient_ids = tuple(patient_ids)
    if not cfg.eval.patient_ids or len(set(cfg.eval.patient_ids)) != len(cfg.eval.patient_ids):
        raise ValueError("Select at least one patient, without duplicates.")
    if any(not re.fullmatch(r"chb\d{2}", pid) for pid in cfg.eval.patient_ids):
        raise ValueError("Patient IDs must have the form chb01.")
    if cfg.feature.epoch_len_s <= 0 or cfg.feature.history_epochs < 0:
        raise ValueError("Epoch length must be positive and history nonnegative.")
    if cfg.eval.top_k_features <= 0 or cfg.eval.min_duration_epochs <= 0:
        raise ValueError("Top-K and minimum duration must be positive.")
    if not cfg.eval.threshold_grid or any(not 0 <= t <= 1 for t in cfg.eval.threshold_grid):
        raise ValueError("Threshold grid must contain probabilities in [0, 1].")
    return cfg
