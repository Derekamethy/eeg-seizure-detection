"""Threshold selection and event-level evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import joblib
import pandas as pd
from scipy.signal import butter, medfilt, sosfiltfilt

from .config import ExperimentConfig
from .data import build_patient_matrix_index, collect_rows_from_matrix_index
from .models import compute_scale_pos_weight, make_model, sample_training_rows, select_top_k_features_tree, topk_feature_cache_hash

def apply_duration_constraint(binary_preds: np.ndarray, min_epochs: int=3) -> np.ndarray:
    cleaned = np.asarray(binary_preds, dtype=np.int8).copy()
    if cleaned.size == 0 or min_epochs <= 1:
        return cleaned
    padded = np.pad(cleaned, (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    short_runs = ends - starts < min_epochs
    for run_start, run_end in zip(starts[short_runs], ends[short_runs]):
        cleaned[run_start:run_end] = 0
    return cleaned

def count_events(labels: np.ndarray) -> int:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0
    return int((labels[0] == 1) + np.sum((labels[1:] == 1) & (labels[:-1] == 0)))

def extract_binary_runs(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0:
        empty = np.empty((0,), dtype=int)
        return (empty, empty)
    padded = np.pad(labels, (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1).astype(int)
    ends = np.flatnonzero(diff == -1).astype(int)
    return (starts, ends)

def compute_event_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray, epoch_len_s: int, recording_lengths: Optional[Sequence[int]]=None) -> Dict[str, Any]:
    """Score overlap with seizure runs; FAR counts non-seizure alarm fragments.

    Divide false alarms by all evaluated hours. Delay is quantized to epoch
    starts, not causal alarm-availability time. Never join recording boundaries.
    """
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    if y_true.ndim != 1 or y_true.shape != y_pred_binary.shape or epoch_len_s <= 0:
        raise ValueError("Aligned one-dimensional labels and positive epoch length required.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred_binary, [0, 1]).all():
        raise ValueError("Event labels must be binary.")
    if recording_lengths is not None:
        slices = recording_slices(len(y_true), recording_lengths)
        parts = [compute_event_metrics(y_true[a:b], y_pred_binary[a:b], epoch_len_s) for a, b in slices]
        events = sum(m['events'] for m in parts)
        detected = sum(m['detected_events'] for m in parts)
        false_alarms = sum(m['false_alarm_events'] for m in parts)
        hours = sum(m['hours'] for m in parts)
        delays = [d for m in parts for d in m['delays_s']]
        return {'events': events, 'detected_events': detected,
                'sensitivity': detected / events if events else np.nan,
                'false_alarm_events': false_alarms,
                'far_per_hour': false_alarms / hours if hours else np.nan,
                'mean_delay_s': float(np.mean(delays)) if delays else np.nan,
                'median_delay_s': float(np.median(delays)) if delays else np.nan,
                'hours': hours, 'delays_s': delays}
    true_starts, true_ends = extract_binary_runs(y_true)
    total_events = int(len(true_starts))
    detected_events = 0
    delays_s = []
    for event_start, event_end in zip(true_starts, true_ends):
        pred_positions = np.flatnonzero(y_pred_binary[event_start:event_end] == 1)
        if pred_positions.size > 0:
            detected_events += 1
            delays_s.append(int(pred_positions[0]) * epoch_len_s)
    false_alarm_mask = ((y_pred_binary == 1) & (y_true == 0)).astype(np.int8)
    false_alarm_events = int(len(extract_binary_runs(false_alarm_mask)[0]))
    total_hours = len(y_true) * epoch_len_s / 3600.0
    sensitivity = detected_events / total_events if total_events > 0 else np.nan
    far_per_hour = false_alarm_events / total_hours if total_hours > 0 else np.nan
    return {'events': float(total_events), 'detected_events': float(detected_events), 'sensitivity': float(sensitivity) if not np.isnan(sensitivity) else np.nan, 'false_alarm_events': float(false_alarm_events), 'far_per_hour': float(far_per_hour) if not np.isnan(far_per_hour) else np.nan, 'mean_delay_s': float(np.mean(delays_s)) if delays_s else np.nan, 'median_delay_s': float(np.median(delays_s)) if delays_s else np.nan, 'hours': float(total_hours), 'delays_s': delays_s}

def recording_slices(size: int, lengths: Sequence[int]) -> List[Tuple[int, int]]:
    if any(int(n) != n or n < 0 for n in lengths) or sum(lengths) != size:
        raise ValueError("Recording lengths must be nonnegative integers summing to vector length.")
    boundaries = np.cumsum([0, *lengths], dtype=int)
    return list(zip(boundaries[:-1], boundaries[1:]))


def smooth_recordings(scores: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Apply the reported centred five-epoch median independently per recording."""
    out = np.asarray(scores, dtype=float).copy()
    for start, end in recording_slices(len(out), lengths):
        if end > start:
            padded = np.pad(out[start:end], (2, 2), constant_values=0)
            out[start:end] = medfilt(padded, kernel_size=5)[2:-2]
    return out


def threshold_recordings(scores, threshold, min_epochs, lengths):
    out = np.zeros(len(scores), dtype=np.int8)
    for start, end in recording_slices(len(scores), lengths):
        out[start:end] = apply_duration_constraint(scores[start:end] >= threshold, min_epochs)
    return out


def choose_threshold_from_validation(y_true: np.ndarray, y_score: np.ndarray, cfg: ExperimentConfig, recording_lengths: Optional[Sequence[int]]=None) -> Dict[str, float]:
    lengths = [len(y_true)] if recording_lengths is None else recording_lengths
    if len(y_true) != len(y_score) or not cfg.eval.threshold_grid:
        raise ValueError("Validation scores must align and threshold grid must not be empty.")
    y_score = smooth_recordings(y_score, lengths)
    best = None
    fallback = None
    threshold_grid = cfg.eval.threshold_grid
    min_duration = cfg.eval.min_duration_epochs
    epoch_len_s = cfg.feature.epoch_len_s
    min_sens = cfg.eval.min_acceptable_sensitivity
    for thr in threshold_grid:
        y_pred = threshold_recordings(y_score, thr, min_duration, lengths)
        metrics = compute_event_metrics(y_true, y_pred, epoch_len_s, lengths)
        row = {'threshold': float(thr), 'sensitivity': metrics['sensitivity'], 'far_per_hour': metrics['far_per_hour'], 'median_delay_s': metrics['median_delay_s']}
        _s = row['sensitivity'] if not np.isnan(row['sensitivity']) else -1.0
        _f = row['far_per_hour'] if not np.isnan(row['far_per_hour']) else float('inf')
        _d = row['median_delay_s'] if not np.isnan(row['median_delay_s']) else float('inf')
        row_rank = (-_s, _f, _d)
        if fallback is None:
            fallback = row
            fallback_rank = row_rank
        elif row_rank < fallback_rank:
            fallback = row
            fallback_rank = row_rank
        if not np.isnan(metrics['sensitivity']) and metrics['sensitivity'] >= min_sens:
            best_row_rank = (_f, _d, -_s)
            if best is None:
                best = row
                best_rank = best_row_rank
            elif best_row_rank < best_rank:
                best = row
                best_rank = best_row_rank
    return best if best is not None else fallback

def split_inner_validation_files(train_seizure_files: List[str], train_bg_files: List[str], seed: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    if len(train_seizure_files) >= 2:
        shuffled_seizures = train_seizure_files.copy()
        rng.shuffle(shuffled_seizures)
        val_seizure_files = [shuffled_seizures[0]]
        inner_train_seizure_files = shuffled_seizures[1:]
    else:
        val_seizure_files = []
        inner_train_seizure_files = train_seizure_files.copy()
    shuffled_bg = train_bg_files.copy()
    rng.shuffle(shuffled_bg)
    n_val_bg = max(1, len(shuffled_bg) // max(2, len(train_seizure_files) + 1)) if len(shuffled_bg) > 0 else 0
    val_bg_files = shuffled_bg[:n_val_bg]
    inner_train_bg_files = shuffled_bg[n_val_bg:]
    return (inner_train_seizure_files, inner_train_bg_files, val_seizure_files, val_bg_files)

def evaluate_patient_loso(patient_payload: Dict[str, Any], cfg: ExperimentConfig, model_name: str='random_forest', top_k: Optional[int]=None, patient_seed_offset: int=0, fold_topk_cache: Optional[Dict[Tuple[Any, ...], np.ndarray]]=None) -> Dict[str, Any]:
    patient_id = patient_payload['meta']['patient_id']
    file_payload = patient_payload['files']
    seizure_files = sorted([f for f, d in file_payload.items() if d['has_seizure']])
    bg_files = sorted([f for f, d in file_payload.items() if not d['has_seizure']])
    if len(seizure_files) < 2:
        raise ValueError('Seizure file count < 2; cannot run LOSO.')
    X_all, y_all, file_row_spans = build_patient_matrix_index(patient_payload)
    feature_cfg_hash = topk_feature_cache_hash(cfg)
    rng = np.random.default_rng(cfg.eval.random_state + patient_seed_offset)
    shuffled_bg = bg_files.copy()
    rng.shuffle(shuffled_bg)
    bg_chunks = np.array_split(shuffled_bg, len(seizure_files))
    fold_rows = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    all_recording_lengths = []
    all_selected_features = []
    topk_cache_hits = 0
    topk_cache_misses = 0
    for outer_idx, test_seizure_file in enumerate(seizure_files):
        outer_train_seizure_files = seizure_files[:outer_idx] + seizure_files[outer_idx + 1:]
        outer_test_bg_files = list(bg_chunks[outer_idx])
        outer_test_bg_set = set(outer_test_bg_files)
        outer_train_bg_files = [f for f in bg_files if f not in outer_test_bg_set]
        inner_train_seizure_files, inner_train_bg_files, val_seizure_files, val_bg_files = split_inner_validation_files(outer_train_seizure_files, outer_train_bg_files, seed=cfg.eval.random_state + patient_seed_offset + outer_idx)
        inner_train_files = inner_train_seizure_files + inner_train_bg_files
        X_inner_raw, y_inner_raw = collect_rows_from_matrix_index(X_all=X_all, y_all=y_all, file_row_spans=file_row_spans, file_names=inner_train_files, feature_indices=None)
        if len(y_inner_raw) == 0:
            raise ValueError('No inner-train samples available in this fold.')
        feature_indices = None
        if top_k is not None:
            selector_seed = cfg.eval.random_state + 1000 + outer_idx
            cache_key = (patient_id, (joblib.hash((X_inner_raw, y_inner_raw)) if fold_topk_cache is not None else None), model_name, cfg.eval.neg_to_pos_ratio, cfg.eval.min_neg_samples, feature_cfg_hash, int(top_k), int(outer_idx), tuple(inner_train_files), int(selector_seed))
            if fold_topk_cache is not None and cache_key in fold_topk_cache:
                feature_indices = fold_topk_cache[cache_key]
                topk_cache_hits += 1
            else:
                selector_rng = np.random.default_rng(selector_seed)
                X_selector, y_selector = sample_training_rows(X_inner_raw, y_inner_raw, cfg, selector_rng)
                selector_model_name = model_name if model_name in {'random_forest', 'xgboost'} else 'random_forest'
                feature_indices = select_top_k_features_tree(X_selector, y_selector, top_k=top_k, cfg=cfg, selector_model_name=selector_model_name)
                if fold_topk_cache is not None:
                    fold_topk_cache[cache_key] = feature_indices
                topk_cache_misses += 1
            all_selected_features.append(feature_indices)
        use_fixed_threshold = bool(getattr(cfg.eval, 'fixed_threshold_mode', False))
        if use_fixed_threshold:
            threshold = float(getattr(cfg.eval, 'fixed_threshold_value', cfg.eval.default_threshold))
        else:
            threshold = cfg.eval.default_threshold
            if len(val_seizure_files) > 0:
                X_train_inner = X_inner_raw if feature_indices is None else X_inner_raw[:, feature_indices]
                train_rng = np.random.default_rng(cfg.eval.random_state + 2000 + outer_idx)
                X_train_inner, y_train_inner = sample_training_rows(X_train_inner, y_inner_raw, cfg, train_rng)
                inner_model = make_model(model_name, cfg, scale_pos_weight=compute_scale_pos_weight(y_train_inner))
                inner_model.fit(X_train_inner, y_train_inner)
                X_val, y_val = collect_rows_from_matrix_index(X_all=X_all, y_all=y_all, file_row_spans=file_row_spans, file_names=val_seizure_files + val_bg_files, feature_indices=feature_indices)
                val_scores = inner_model.predict_proba(X_val)[:, 1]
                threshold_info = choose_threshold_from_validation(y_val, val_scores, cfg, [len(file_payload[f]['y']) for f in val_seizure_files + val_bg_files])
                threshold = threshold_info['threshold']
        X_train_outer, y_train_outer = collect_rows_from_matrix_index(X_all=X_all, y_all=y_all, file_row_spans=file_row_spans, file_names=outer_train_seizure_files + outer_train_bg_files, feature_indices=feature_indices)
        outer_rng = np.random.default_rng(cfg.eval.random_state + 3000 + outer_idx)
        X_train_outer, y_train_outer = sample_training_rows(X_train_outer, y_train_outer, cfg, outer_rng)
        model = make_model(model_name, cfg, scale_pos_weight=compute_scale_pos_weight(y_train_outer))
        model.fit(X_train_outer, y_train_outer)
        X_test, y_test = collect_rows_from_matrix_index(X_all=X_all, y_all=y_all, file_row_spans=file_row_spans, file_names=sorted([test_seizure_file] + outer_test_bg_files), feature_indices=feature_indices)
        y_score = model.predict_proba(X_test)[:, 1]
        lengths = [len(file_payload[f]['y']) for f in sorted([test_seizure_file] + outer_test_bg_files)]
        all_recording_lengths.extend(lengths)
        y_score_smoothed = smooth_recordings(y_score, lengths)
        y_pred = threshold_recordings(y_score_smoothed, threshold, cfg.eval.min_duration_epochs, lengths)
        fold_metrics = compute_event_metrics(y_test, y_pred, cfg.feature.epoch_len_s, lengths)
        fold_rows.append({'fold': outer_idx, 'threshold': threshold, 'test_seizure_file': test_seizure_file, 'n_test_bg_files': len(outer_test_bg_files), 'sensitivity': fold_metrics['sensitivity'], 'far_per_hour': fold_metrics['far_per_hour'], 'median_delay_s': fold_metrics['median_delay_s']})
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_score.append(y_score_smoothed)
    y_true_cat = np.concatenate(all_y_true)
    y_pred_cat = np.concatenate(all_y_pred)
    y_score_cat = np.concatenate(all_y_score)
    patient_metrics = compute_event_metrics(y_true_cat, y_pred_cat, cfg.feature.epoch_len_s, all_recording_lengths)
    summary = {'Patient': patient_id, 'Model': model_name, 'TopK': top_k if top_k is not None else -1, 'Hours': patient_metrics['hours'], 'True_Seizures': patient_metrics['events'], 'Detected_Seizures': patient_metrics['detected_events'], 'False_Alarm_Events': patient_metrics['false_alarm_events'], 'Sensitivity': patient_metrics['sensitivity'], 'FAR_per_Hour': patient_metrics['far_per_hour'], 'Mean_Delay_s': patient_metrics['mean_delay_s'], 'Median_Delay_s': patient_metrics['median_delay_s'], 'Median_Threshold': float(np.median([row['threshold'] for row in fold_rows]))}
    return {'summary': summary, 'folds': pd.DataFrame(fold_rows), 'y_true': y_true_cat, 'y_pred': y_pred_cat, 'y_score': y_score_cat, 'recording_lengths': all_recording_lengths, 'selected_features': all_selected_features, 'topk_cache_hits': int(topk_cache_hits), 'topk_cache_misses': int(topk_cache_misses)}

def evaluate_many_patients(caches: Dict[str, Dict[str, Any]], cfg: ExperimentConfig, model_name: str='random_forest', top_k: Optional[int]=None, fold_topk_cache: Optional[Dict[Tuple[Any, ...], np.ndarray]]=None) -> Dict[str, Any]:
    patient_outputs = {}
    summary_rows = []
    total_cache_hits = 0
    total_cache_misses = 0
    for patient_idx, patient_id in enumerate(cfg.eval.patient_ids):
        if patient_id not in caches:
            raise ValueError(f'Missing cache for requested patient: {patient_id}')
        try:
            result = evaluate_patient_loso(patient_payload=caches[patient_id], cfg=cfg, model_name=model_name, top_k=top_k, patient_seed_offset=patient_idx * 100, fold_topk_cache=fold_topk_cache)
            patient_outputs[patient_id] = result
            summary_rows.append(result['summary'])
            total_cache_hits += int(result.get('topk_cache_hits', 0))
            total_cache_misses += int(result.get('topk_cache_misses', 0))
            print(f"[{patient_id}] {model_name} | Sens={result['summary']['Sensitivity']:.2%} | FAR/hr={result['summary']['FAR_per_Hour']:.4f} | MedianThr={result['summary']['Median_Threshold']:.3f}")
        except Exception as exc:
            raise RuntimeError(f'{patient_id} evaluation failed: {exc}') from exc
    summary_df = pd.DataFrame(summary_rows)
    return {'patient_outputs': patient_outputs, 'summary_df': summary_df, 'topk_cache_hits': int(total_cache_hits), 'topk_cache_misses': int(total_cache_misses)}

def build_event_detection_matrix(result_bundle: Dict[str, Any], cfg: ExperimentConfig, label: str) -> pd.DataFrame:
    patient_outputs = result_bundle.get('patient_outputs', {})
    tp_events = 0.0
    fn_events = 0.0
    fp_events = 0.0
    total_hours = 0.0
    delay_weighted_sum = 0.0
    delay_weight = 0.0
    for _, patient_res in patient_outputs.items():
        metrics = compute_event_metrics(patient_res['y_true'], patient_res['y_pred'], cfg.feature.epoch_len_s, patient_res['recording_lengths'])
        events = float(metrics['events'])
        detected = float(metrics['detected_events'])
        false_alarms = float(metrics['false_alarm_events'])
        hours = float(metrics['hours'])
        tp_events += detected
        fn_events += max(0.0, events - detected)
        fp_events += false_alarms
        total_hours += hours
        if not np.isnan(metrics['mean_delay_s']) and detected > 0:
            delay_weighted_sum += float(metrics['mean_delay_s']) * detected
            delay_weight += detected
    sensitivity = tp_events / max(1.0, tp_events + fn_events)
    far_per_hour = fp_events / max(1e-09, total_hours)
    mean_delay_s = delay_weighted_sum / delay_weight if delay_weight > 0 else np.nan
    return pd.DataFrame([{'Variant': label, 'TP_events': float(tp_events), 'FN_events': float(fn_events), 'FP_events': float(fp_events), 'Sensitivity': float(sensitivity), 'FAR_per_Hour': float(far_per_hour), 'Mean_Delay_s': float(mean_delay_s) if not np.isnan(mean_delay_s) else np.nan, 'Hours': float(total_hours), 'Patients': int(len(patient_outputs))}])
