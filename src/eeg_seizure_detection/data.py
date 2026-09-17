"""Patient-cache construction and matrix assembly utilities."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import joblib
import mne
import numpy as np
import pandas as pd

from .config import CACHE_SCHEMA_VERSION, ExperimentConfig
from .features import build_base_feature_names, build_stacked_feature_names, extract_base_features_for_file, temporal_stack_features
from .io_utils import config_to_hash, ensure_dir, get_patient_cache_path
from .preprocessing import align_channels, bandpass_filter_multich, build_epoch_labels, deduplicate_and_normalize_raw

def parse_summary_to_dict(summary_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a CHB-MIT patient summary into EDF seizure intervals.

    Supports standard and numbered seizure start/end fields as well as
    EDF filenames containing characters such as ``+``.
    """
    seizure_dict: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None
    current_start: Optional[int] = None
    expected_counts = {}
    if not summary_path.exists():
        raise FileNotFoundError(f'Seizure summary not found: {summary_path}')
    file_pattern = re.compile('File Name:\\s*([^\\s]+\\.edf)', flags=re.IGNORECASE)
    start_pattern = re.compile('Seizure(?:\\s+\\d+)?\\s+Start Time:\\s*(\\d+)\\s*(?:seconds?)?', flags=re.IGNORECASE)
    end_pattern = re.compile('Seizure(?:\\s+\\d+)?\\s+End Time:\\s*(\\d+)\\s*(?:seconds?)?', flags=re.IGNORECASE)
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            file_match = file_pattern.search(line)
            if file_match:
                if current_start is not None:
                    raise ValueError('Unpaired seizure start before next recording.')
                current_file = Path(file_match.group(1)).name
                if current_file in seizure_dict:
                    raise ValueError(f'Duplicate summary entry: {current_file}')
                seizure_dict[current_file] = []
                current_start = None
                continue
            count_match = re.search(r'Number of Seizures in File:\s*(\d+)', line, re.IGNORECASE)
            if count_match and current_file is not None:
                expected_counts[current_file] = int(count_match.group(1))
            start_match = start_pattern.search(line)
            if start_match:
                if current_start is not None:
                    raise ValueError('Consecutive seizure starts without an end.')
                current_start = int(start_match.group(1))
                continue
            end_match = end_pattern.search(line)
            if end_match:
                if current_file is None or current_start is None:
                    raise ValueError('Seizure end without a corresponding start.')
                current_end = int(end_match.group(1))
                if current_end <= current_start:
                    raise ValueError('Seizure end must follow its start.')
                seizure_dict[current_file].append((current_start, current_end))
                current_start = None
    if current_start is not None:
        raise ValueError('Unpaired seizure start in summary.')
    for name, intervals in seizure_dict.items():
        if name not in expected_counts or expected_counts[name] != len(intervals):
            raise ValueError(f'Seizure count missing or inconsistent for {name}.')
    return seizure_dict

def build_patient_cache(cfg: ExperimentConfig, patient_id: str, overwrite: bool=False) -> Path:
    patient_dir = Path(cfg.data_root) / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f'Patient directory not found: {patient_dir}')
    cache_path = get_patient_cache_path(cfg, patient_id)
    if cache_path.exists() and (not overwrite):
        print(f'[cache] {patient_id}: using existing compatible cache -> {cache_path.name}')
        return cache_path
    summary_path = patient_dir / f'{patient_id}-summary.txt'
    seizure_dict = parse_summary_to_dict(summary_path)
    edf_paths = sorted(patient_dir.glob('*.edf'))
    if not edf_paths:
        raise ValueError(f'No EDF recordings in {patient_dir}')
    missing = [p.name for p in edf_paths if p.name not in seizure_dict]
    if missing:
        raise ValueError(f'Recordings missing from seizure summary: {missing}')
    absent_edfs = sorted(set(seizure_dict) - {p.name for p in edf_paths})
    if absent_edfs:
        raise FileNotFoundError(f'Annotated recordings missing from patient directory: {absent_edfs}')
    file_payload = {}
    processed_files = 0
    skipped_files = 0
    for edf_path in edf_paths:
        raw = None
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            raw = deduplicate_and_normalize_raw(raw)
            aligned_data, align_info = align_channels(raw=raw, target_channels=cfg.channels, policy=cfg.feature.channel_missing_policy)
            if cfg.feature.scale_to_uV:
                aligned_data = aligned_data * 1000000.0
            fs = int(raw.info['sfreq'])
            if fs != raw.info['sfreq']:
                raise ValueError('Sampling frequency must be an integer number of Hz.')
            data_bp = bandpass_filter_multich(aligned_data, fs=fs, lowcut=cfg.feature.bandpass_low_hz, highcut=cfg.feature.bandpass_high_hz, method=cfg.feature.bandpass_method, butter_order=cfg.feature.butter_order)
            X_base, kept_epoch_indices = extract_base_features_for_file(data_bp, fs, cfg)
            if X_base.size == 0:
                raise ValueError('Recording contains no complete epochs.')
            seizure_intervals = seizure_dict.get(edf_path.name, [])
            y_base = build_epoch_labels(n_epochs=len(kept_epoch_indices), epoch_len_s=cfg.feature.epoch_len_s, seizure_intervals=seizure_intervals)
            X_stacked, y_stacked = temporal_stack_features(X_base=X_base, y=y_base, history_epochs=cfg.feature.history_epochs)
            file_payload[edf_path.name] = {'X': X_stacked.astype(np.float32), 'y': y_stacked.astype(np.int8), 'has_seizure': bool(len(seizure_intervals) > 0), 'has_positive_epoch': bool(np.any(y_stacked == 1)), 'fs': fs, 'n_epochs': int(len(y_stacked)), 'missing_channels': align_info['missing_channels'], 'missing_count': align_info['missing_count'], 'reversed_channels': align_info['reversed_channels']}
            processed_files += 1
        except Exception as exc:
            skipped_files += 1
            raise RuntimeError(f'{patient_id}/{edf_path.name}: {exc}') from exc
        finally:
            if raw is not None:
                raw.close()
    payload = {'meta': {'patient_id': patient_id, 'config_hash': config_to_hash(cfg), 'base_feature_dim': len(build_base_feature_names(cfg)), 'stacked_feature_dim': len(build_stacked_feature_names(cfg)), 'channel_missing_policy': cfg.feature.channel_missing_policy, 'history_epochs': cfg.feature.history_epochs, 'epoch_len_s': cfg.feature.epoch_len_s, 'cache_schema_version': CACHE_SCHEMA_VERSION, 'processed_files': processed_files, 'skipped_files': skipped_files}, 'files': file_payload}
    ensure_dir(cache_path.parent)
    joblib.dump(payload, cache_path)
    print(f'[cache] {patient_id}: processed={processed_files}, skipped={skipped_files}, file={cache_path.name}')
    return cache_path

def build_all_caches(cfg: ExperimentConfig, overwrite: bool=False, n_jobs: int=1) -> List[Path]:
    if not Path(cfg.data_root).exists():
        raise FileNotFoundError(f'Current data_root does not exist: {cfg.data_root}\nPlease set CFG.data_root to your local CHB-MIT directory.')
    n_jobs = int(max(1, n_jobs))
    patient_ids = list(cfg.eval.patient_ids)
    if n_jobs == 1:
        cache_paths = []
        for patient_id in patient_ids:
            cache_paths.append(build_patient_cache(cfg, patient_id, overwrite=overwrite))
            gc.collect()
        return cache_paths
    print(f'Building caches in parallel: n_jobs={n_jobs}, patients={len(patient_ids)}')
    cache_paths = joblib.Parallel(n_jobs=n_jobs, backend='loky')((joblib.delayed(build_patient_cache)(cfg, patient_id, overwrite=overwrite) for patient_id in patient_ids))
    return list(cache_paths)

def load_patient_cache(cfg: ExperimentConfig, patient_id: str) -> Dict[str, Any]:
    cache_path = get_patient_cache_path(cfg, patient_id)
    if not cache_path.exists():
        raise FileNotFoundError(f'Cache file not found: {cache_path}')
    payload = joblib.load(cache_path)
    meta = payload.get('meta', {})
    files = payload.get('files', {})
    expected_hash = config_to_hash(cfg)
    if meta.get('config_hash') != expected_hash:
        raise ValueError(f"Cache configuration mismatch for {patient_id}.\nCached hash = {meta.get('config_hash')}\nCurrent hash = {expected_hash}\nRebuild the patient cache with the current configuration.")
    if not isinstance(files, dict):
        raise ValueError(f"Invalid cache structure for {patient_id}: missing 'files' mapping.")
    return payload

def load_all_caches(cfg: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    caches = {}
    for patient_id in cfg.eval.patient_ids:
        cache_path = get_patient_cache_path(cfg, patient_id)
        if not cache_path.exists():
            raise FileNotFoundError(f'Missing cache for {patient_id}; run with --rebuild-cache.')
        caches[patient_id] = load_patient_cache(cfg, patient_id)
    return caches

def summarize_caches(caches: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for patient_id, payload in caches.items():
        files = payload['files']
        n_files = len(files)
        n_seizure_files = sum((int(d['has_seizure']) for d in files.values()))
        n_positive_epoch_files = sum((int(d.get('has_positive_epoch', d['has_seizure'])) for d in files.values()))
        n_epochs = sum((int(d['n_epochs']) for d in files.values()))
        rows.append({'Patient': patient_id, 'Files': n_files, 'Seizure_Files': n_seizure_files, 'Positive_Epoch_Files': n_positive_epoch_files, 'Epochs': n_epochs, 'Feature_Dim': payload['meta']['stacked_feature_dim']})
    return pd.DataFrame(rows).sort_values('Patient').reset_index(drop=True)

def build_patient_matrix_index(patient_payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    cache_key = '_matrix_index_cache'
    cached = patient_payload.get(cache_key)
    if cached is not None:
        return (cached['X_all'], cached['y_all'], cached['file_row_spans'])
    file_payload = patient_payload['files']
    ordered_files = sorted(file_payload.keys())
    X_blocks = []
    y_blocks = []
    file_row_spans = {}
    row_start = 0
    for file_name in ordered_files:
        item = file_payload[file_name]
        X_block = np.asarray(item['X'], dtype=np.float32)
        y_block = np.asarray(item['y'], dtype=np.int8)
        row_end = row_start + len(y_block)
        file_row_spans[file_name] = (row_start, row_end)
        X_blocks.append(X_block)
        y_blocks.append(y_block)
        row_start = row_end
    if len(X_blocks) == 0:
        X_all = np.empty((0, 0), dtype=np.float32)
        y_all = np.empty((0,), dtype=np.int8)
    else:
        X_all = np.vstack(X_blocks).astype(np.float32, copy=False)
        y_all = np.concatenate(y_blocks).astype(np.int8, copy=False)
    patient_payload[cache_key] = {'X_all': X_all, 'y_all': y_all, 'file_row_spans': file_row_spans}
    return (X_all, y_all, file_row_spans)

def collect_rows_from_matrix_index(X_all: np.ndarray, y_all: np.ndarray, file_row_spans: Dict[str, Tuple[int, int]], file_names: Sequence[str], feature_indices: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    if len(file_names) == 0:
        feature_dim = X_all.shape[1] if feature_indices is None else int(len(feature_indices))
        return (np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int8))
    blocks_X = []
    blocks_y = []
    for file_name in file_names:
        if file_name not in file_row_spans:
            raise KeyError(f'Unknown file in row index: {file_name}')
        start, end = file_row_spans[file_name]
        X_block = X_all[start:end]
        if feature_indices is not None:
            X_block = X_block[:, feature_indices]
        blocks_X.append(X_block)
        blocks_y.append(y_all[start:end])
    if len(blocks_X) == 1:
        return (blocks_X[0], blocks_y[0])
    return (np.concatenate(blocks_X, axis=0), np.concatenate(blocks_y, axis=0))

def collect_rows_from_files(patient_payload: Dict[str, Any], file_names: Sequence[str], feature_indices: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper; callers with existing matrix index should use
    collect_rows_from_matrix_index directly to avoid redundant lookups (optimization #20)."""
    X_all, y_all, file_row_spans = build_patient_matrix_index(patient_payload)
    return collect_rows_from_matrix_index(X_all=X_all, y_all=y_all, file_row_spans=file_row_spans, file_names=file_names, feature_indices=feature_indices)
