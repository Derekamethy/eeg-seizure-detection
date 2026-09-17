"""Spectral, synchrony and temporal feature engineering."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from .config import ExperimentConfig
from .preprocessing import normalize_channel_name

_BAND_MASK_CACHE: Dict[Tuple[int, int, Tuple[int, ...]], List[np.ndarray]] = {}

_SYNC_INDEX_CACHE: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], List[Tuple[int, int]]] = {}

def get_band_tuples(cfg: ExperimentConfig) -> List[Tuple[str, Tuple[int, int]]]:
    edges = cfg.feature.band_edges_hz
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        bands.append((f'{low}-{high}Hz', (int(low), int(high))))
    return bands

def get_synchrony_index_pairs(cfg: ExperimentConfig) -> List[Tuple[int, int]]:
    channels_norm = tuple((normalize_channel_name(ch) for ch in cfg.channels))
    pairs_norm = tuple(((normalize_channel_name(a), normalize_channel_name(b)) for a, b in cfg.feature.synchrony_pairs_names))
    key = (channels_norm, pairs_norm)
    cached = _SYNC_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    channel_to_idx = {ch: i for i, ch in enumerate(channels_norm)}
    pairs = []
    for left_name, right_name in pairs_norm:
        pairs.append((channel_to_idx[left_name], channel_to_idx[right_name]))
    _SYNC_INDEX_CACHE[key] = pairs
    return pairs

def _get_band_masks(fs: int, epoch_samples: int, band_edges: Sequence[int]) -> List[np.ndarray]:
    edges = tuple((int(x) for x in band_edges))
    key = (int(fs), int(epoch_samples), edges)
    cached = _BAND_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    freqs = np.fft.rfftfreq(epoch_samples, d=1.0 / fs)
    masks = []
    for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = idx == len(edges) - 2
        if is_last:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        masks.append(mask)
    _BAND_MASK_CACHE[key] = masks
    return masks

def extract_base_features_for_file(data_bp: np.ndarray, fs: int, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    epoch_len_s = cfg.feature.epoch_len_s
    epoch_samples = epoch_len_s * fs
    n_epochs = data_bp.shape[1] // epoch_samples
    if n_epochs == 0:
        return (np.empty((0, 0), dtype=float), np.empty((0,), dtype=int))
    usable = data_bp[:, :n_epochs * epoch_samples]
    epochs = usable.reshape(data_bp.shape[0], n_epochs, epoch_samples).transpose(1, 0, 2)
    epochs_centered = epochs - epochs.mean(axis=2, keepdims=True)
    fft_vals = np.fft.rfft(epochs_centered, axis=-1)
    power = fft_vals.real ** 2 + fft_vals.imag ** 2
    band_masks = _get_band_masks(fs, epoch_samples, cfg.feature.band_edges_hz)
    band_features = [power[:, :, mask].sum(axis=-1) for mask in band_masks]
    band_cube = np.stack(band_features, axis=-1)
    band_flat = band_cube.reshape(n_epochs, -1)
    sync_values = []
    for idx_a, idx_b in get_synchrony_index_pairs(cfg):
        sig_a = epochs_centered[:, idx_a, :]
        sig_b = epochs_centered[:, idx_b, :]
        numerator = np.sum(sig_a * sig_b, axis=1)
        denominator = np.sqrt(np.sum(sig_a ** 2, axis=1) * np.sum(sig_b ** 2, axis=1))
        corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        sync_values.append(corr)
    sync_matrix = np.stack(sync_values, axis=1) if sync_values else np.empty((n_epochs, 0), dtype=float)
    X_base = np.hstack([band_flat, sync_matrix]).astype(np.float32)
    return (X_base, np.arange(n_epochs, dtype=int))

def temporal_stack_features(X_base: np.ndarray, y: np.ndarray, history_epochs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-allocate output matrix instead of vstack + hstack (optimization #4)."""
    if X_base.shape[0] != len(y):
        raise ValueError('X_base and y length mismatch.')
    if history_epochs < 0:
        raise ValueError('history_epochs must be nonnegative.')
    n_epochs, base_dim = X_base.shape
    total_dim = base_dim * (history_epochs + 1)
    X_stacked = np.zeros((n_epochs, total_dim), dtype=np.float32)
    col = 0
    for lag in range(history_epochs, -1, -1):
        dst = X_stacked[:, col:col + base_dim]
        if lag == 0:
            dst[:] = X_base
        else:
            dst[lag:] = X_base[:-lag]
        col += base_dim
    return (X_stacked, y.copy())


def build_base_feature_names(cfg: ExperimentConfig) -> List[str]:
    names = []
    bands = get_band_tuples(cfg)
    for ch in cfg.channels:
        for band_name, _ in bands:
            names.append(f'{ch} [{band_name}]')
    for idx_a, idx_b in get_synchrony_index_pairs(cfg):
        names.append(f'Sync: {cfg.channels[idx_a]} & {cfg.channels[idx_b]}')
    return names


def build_stacked_feature_names(cfg: ExperimentConfig) -> List[str]:
    base_names = build_base_feature_names(cfg)
    names = []
    history = cfg.feature.history_epochs
    for lag in range(history, 0, -1):
        for name in base_names:
            names.append(f'{name} @ t-{lag}')
    for name in base_names:
        names.append(f'{name} @ t')
    return names
