"""Benchmark-table and experiment-summary helpers."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .config import CFG, ExperimentConfig
from .data import load_all_caches
from .evaluation import evaluate_many_patients

def _safe_float(x, default=0.0):
    """Simplified: single try/except block (optimization #15)."""
    try:
        return float(x) if x is not None and (not np.isnan(x)) else float(default)
    except (TypeError, ValueError):
        return float(default)

def _aggregate_summary(summary_df: pd.DataFrame) -> Dict[str, float]:
    if summary_df is None or summary_df.empty:
        return {'mean_sensitivity': np.nan, 'median_far_per_hour': np.nan, 'mean_delay_s': np.nan, 'patients': 0}
    return {'mean_sensitivity': float(summary_df['Sensitivity'].mean()), 'median_far_per_hour': float(summary_df['FAR_per_Hour'].median()), 'mean_delay_s': float(summary_df['Mean_Delay_s'].mean()), 'patients': int(len(summary_df))}

def _priority_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    work['_sens'] = work['mean_sensitivity'].apply(lambda v: _safe_float(v, -1.0))
    work['_far'] = work['median_far_per_hour'].apply(lambda v: _safe_float(v, 1000000000.0))
    work['_delay'] = work['mean_delay_s'].apply(lambda v: _safe_float(v, 1000000000.0))
    work = work.sort_values(['_sens', '_far', '_delay'], ascending=[False, True, True]).reset_index(drop=True)
    work['priority_rank'] = np.arange(1, len(work) + 1)
    work = work.drop(columns=['_sens', '_far', '_delay'])
    return work

def _run_rf_eval(cfg_variant: ExperimentConfig, caches_variant: Dict[str, Dict[str, Any]], top_k: int, fold_topk_cache: Optional[Dict[Tuple[Any, ...], np.ndarray]]=None):
    t0 = time.perf_counter()
    result = evaluate_many_patients(caches=caches_variant, cfg=cfg_variant, model_name='random_forest', top_k=top_k, fold_topk_cache=fold_topk_cache)
    summary_df = result.get('summary_df', pd.DataFrame())
    agg = _aggregate_summary(summary_df)
    agg['elapsed_s'] = float(time.perf_counter() - t0)
    agg['fold_topk_cache_entries'] = int(len(fold_topk_cache)) if fold_topk_cache is not None else 0
    agg['topk_cache_hits'] = int(result.get('topk_cache_hits', 0))
    agg['topk_cache_misses'] = int(result.get('topk_cache_misses', 0))
    return (result, agg)

def _build_macro_row(summary_df: pd.DataFrame, model_name: str, top_k: Optional[int]) -> Dict[str, Any]:
    if summary_df is None or summary_df.empty:
        return {'Model': model_name, 'TopK': top_k if top_k is not None else -1, 'Patients': 0, 'Sensitivity': np.nan, 'FAR_per_Hour': np.nan, 'Mean_Delay_s': np.nan, 'Median_Delay_s': np.nan, 'Median_Threshold': np.nan}
    return {'Model': model_name, 'TopK': top_k if top_k is not None else -1, 'Patients': int(len(summary_df)), 'Sensitivity': float(summary_df['Sensitivity'].mean()), 'FAR_per_Hour': float(summary_df['FAR_per_Hour'].median()), 'Mean_Delay_s': float(summary_df['Mean_Delay_s'].mean()), 'Median_Delay_s': float(summary_df['Median_Delay_s'].median()), 'Median_Threshold': float(summary_df['Median_Threshold'].median())}

def _sort_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out = out.sort_values(['Sensitivity', 'FAR_per_Hour', 'Mean_Delay_s'], ascending=[False, True, True]).reset_index(drop=True)
    return out

def annotate_bars(ax, fmt='{:.3f}', offset=4):
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (patch.get_x() + patch.get_width() / 2, h), ha='center', va='bottom', fontsize=10, xytext=(0, offset), textcoords='offset points')

def _reorder_cols(df: pd.DataFrame, preferred_cols: list) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols_exist = [c for c in preferred_cols if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_exist]
    return df[cols_exist + cols_rest].copy()

def _add_metric_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if {'Sensitivity', 'FAR_per_Hour', 'Mean_Delay_s'}.issubset(out.columns):
        out = out.sort_values(['Sensitivity', 'FAR_per_Hour', 'Mean_Delay_s'], ascending=[False, True, True]).reset_index(drop=True)
        out.insert(0, 'Rank', np.arange(1, len(out) + 1))
    return out
