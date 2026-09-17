"""Classifier construction, feature selection and patient-specific training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ExperimentConfig
from .data import collect_rows_from_files
from .io_utils import make_json_safe
from .features import build_stacked_feature_names

def sample_training_rows(X: np.ndarray, y: np.ndarray, cfg: ExperimentConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized: single index array avoids multiple data copies (optimization #12)."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0:
        return (X.copy(), y.copy())
    target_neg = max(cfg.eval.min_neg_samples, len(pos_idx) * cfg.eval.neg_to_pos_ratio)
    target_neg = min(target_neg, len(neg_idx))
    if target_neg < len(neg_idx):
        neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
    combined = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(combined)
    return (X[combined], y[combined])

def make_model(model_name: str, cfg: ExperimentConfig, scale_pos_weight: float=1.0):
    if model_name == 'svm_rbf':
        return Pipeline(steps=[('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=cfg.eval.svm_c, gamma=cfg.eval.svm_gamma, probability=True, class_weight='balanced', random_state=cfg.eval.random_state))])
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=cfg.eval.rf_n_estimators, max_depth=cfg.eval.rf_max_depth, min_samples_leaf=cfg.eval.rf_min_samples_leaf, max_features=cfg.eval.rf_max_features, class_weight='balanced_subsample', n_jobs=cfg.eval.rf_n_jobs, random_state=cfg.eval.random_state)
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=cfg.eval.xgb_n_estimators, max_depth=cfg.eval.xgb_max_depth, learning_rate=cfg.eval.xgb_learning_rate, subsample=cfg.eval.xgb_subsample, colsample_bytree=cfg.eval.xgb_colsample_bytree, reg_lambda=cfg.eval.xgb_reg_lambda, min_child_weight=cfg.eval.xgb_min_child_weight, gamma=cfg.eval.xgb_gamma, scale_pos_weight=scale_pos_weight, eval_metric='logloss', tree_method=cfg.eval.xgb_tree_method, n_jobs=cfg.eval.xgb_n_jobs, random_state=cfg.eval.random_state)
    raise ValueError(f'Unsupported model_name: {model_name}')

def compute_scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return float(neg / max(1, pos))

def topk_feature_cache_hash(cfg: ExperimentConfig) -> str:
    payload = {'feature': make_json_safe(asdict(cfg.feature)), 'selector_n_estimators': cfg.eval.selector_n_estimators, 'selector_max_depth': cfg.eval.selector_max_depth, 'rf_max_features': cfg.eval.rf_max_features, 'rf_min_samples_leaf': cfg.eval.rf_min_samples_leaf, 'xgb_tree_method': cfg.eval.xgb_tree_method, 'xgb_n_jobs': cfg.eval.xgb_n_jobs, 'random_state': cfg.eval.random_state}
    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return joblib.hash(payload_str)

def select_top_k_features_tree(X_train: np.ndarray, y_train: np.ndarray, top_k: int, cfg: ExperimentConfig, selector_model_name: str='random_forest') -> np.ndarray:
    if top_k <= 0:
        raise ValueError('top_k must be positive.')
    if X_train.shape[1] <= top_k:
        return np.arange(X_train.shape[1], dtype=int)
    if selector_model_name == 'xgboost':
        from xgboost import XGBClassifier
        scale_pos_weight = compute_scale_pos_weight(y_train)
        selector = XGBClassifier(n_estimators=cfg.eval.selector_n_estimators, max_depth=cfg.eval.selector_max_depth, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight, eval_metric='logloss', tree_method=cfg.eval.xgb_tree_method, n_jobs=cfg.eval.xgb_n_jobs, random_state=cfg.eval.random_state)
    else:
        selector = RandomForestClassifier(n_estimators=cfg.eval.selector_n_estimators, max_depth=cfg.eval.selector_max_depth, min_samples_leaf=cfg.eval.rf_min_samples_leaf, max_features=cfg.eval.rf_max_features, class_weight='balanced_subsample', n_jobs=cfg.eval.rf_n_jobs, random_state=cfg.eval.random_state)
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    top_idx = np.argsort(importances)[-top_k:]
    return np.sort(top_idx)

def train_patient_bundle(patient_id: str, payload: Dict[str, Any], cfg: ExperimentConfig, threshold: float) -> Dict[str, Any]:
    file_names = list(payload['files'].keys())
    X_full, y_full = collect_rows_from_files(payload, file_names, feature_indices=None)
    if len(y_full) == 0:
        raise ValueError(f'No rows for {patient_id}')
    if set(np.unique(y_full)) != {0, 1}:
        raise ValueError("Training requires both seizure and background epochs.")
    all_feature_names = build_stacked_feature_names(cfg)
    if X_full.shape[1] != len(all_feature_names):
        raise ValueError("Feature matrix does not match configuration.")
    rng = np.random.default_rng(cfg.eval.random_state)
    X_sel, y_sel = sample_training_rows(X_full, y_full, cfg, rng)
    top_idx = select_top_k_features_tree(X_sel, y_sel, cfg.eval.top_k_features, cfg)
    X_light = X_full[:, top_idx]
    model = make_model('random_forest', cfg)
    model.fit(X_light, y_full)
    return {'patient_id': patient_id, 'model_name': 'random_forest', 'patient_model_family': 'random_forest_patient_specific', 'model': model, 'top_k_indices': top_idx, 'feature_names': [all_feature_names[int(i)] for i in top_idx], 'threshold': float(threshold), 'requested_top_k': int(cfg.eval.top_k_features), 'input_dim': int(len(top_idx)), 'train_rows': int(len(y_full)), 'train_pos': int(np.sum(y_full == 1)), 'train_neg': int(np.sum(y_full == 0))}
