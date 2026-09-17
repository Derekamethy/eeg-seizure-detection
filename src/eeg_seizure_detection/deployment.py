"""Runtime and model-footprint profiling helpers."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import joblib
import numpy as np

def benchmark_single_epoch_latency(model, X_stream: np.ndarray, n_steps: int=1000) -> Dict[str, float]:
    n_steps = min(n_steps, len(X_stream))
    if n_steps <= 0:
        raise ValueError('Latency profiling requires at least one sample.')
    timings = []
    for i in range(n_steps):
        sample = X_stream[i:i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(sample)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    timings = np.asarray(timings)
    return {'n_steps': int(n_steps), 'mean_ms': float(np.mean(timings) * 1000.0), 'median_ms': float(np.median(timings) * 1000.0), 'p95_ms': float(np.percentile(timings, 95) * 1000.0), 'fps_estimate': float(1.0 / np.mean(timings)) if np.mean(timings) > 0 else np.nan}

def profile_rf_model(model, input_dim: int) -> Dict[str, float]:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    raw_bytes = buffer.getvalue()
    estimators = list(getattr(model, 'estimators_', []))
    total_nodes = int(sum((est.tree_.node_count for est in estimators))) if estimators else 0
    total_leaves = int(sum((np.sum(est.tree_.children_left == -1) for est in estimators))) if estimators else 0
    max_depth = int(max((est.tree_.max_depth for est in estimators))) if estimators else 0
    mean_depth = float(np.mean([est.tree_.max_depth for est in estimators])) if estimators else 0.0
    return {'input_dim': int(input_dim), 'memory_mb': float(len(raw_bytes) / (1024.0 * 1024.0)), 'n_trees': int(len(estimators)), 'total_nodes': int(total_nodes), 'total_leaves': int(total_leaves), 'decision_nodes': int(total_nodes - total_leaves), 'max_tree_depth': int(max_depth), 'mean_tree_depth': float(mean_depth)}
