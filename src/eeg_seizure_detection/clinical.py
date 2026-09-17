"""Case-level inspection helpers for retrospective error analysis."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .config import CFG

def build_clinical_case_dataframe(patient_id: str, patient_payload: Dict[str, Any], model, feature_indices: np.ndarray, threshold: float, pre_seconds: int=12, post_seconds: int=14) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    seizure_files = [f for f, d in patient_payload['files'].items() if d['has_seizure'] and d.get('has_positive_epoch', False)]
    if not seizure_files:
        seizure_files = [f for f, d in patient_payload['files'].items() if d['has_seizure']]
    if not seizure_files:
        raise ValueError(f'No seizure file found for {patient_id}.')
    file_name = sorted(seizure_files)[0]
    item = patient_payload['files'][file_name]
    X = item['X'][:, feature_indices]
    y = item['y'].astype(np.int8)
    seizure_positions = np.flatnonzero(y == 1)
    if len(seizure_positions) == 0:
        raise ValueError(f'No positive epochs in {patient_id}/{file_name}.')
    onset_epoch = int(seizure_positions[0])
    epoch_len_s = patient_payload['meta']['epoch_len_s']
    onset_time_s = onset_epoch * epoch_len_s
    pre_epochs = max(1, int(np.ceil(pre_seconds / epoch_len_s)))
    post_epochs = max(1, int(np.ceil(post_seconds / epoch_len_s)))
    start_epoch = max(0, onset_epoch - pre_epochs)
    end_epoch = min(len(y), onset_epoch + post_epochs + 1)
    epoch_indices = np.arange(start_epoch, end_epoch)
    probs = model.predict_proba(X[start_epoch:end_epoch])[:, 1]
    alarms = probs >= threshold
    case_df = pd.DataFrame({'epoch_index': epoch_indices, 'time_s': epoch_indices * epoch_len_s, 'time_rel_s': (epoch_indices - onset_epoch) * epoch_len_s, 'y_true': y[start_epoch:end_epoch], 'prob': probs, 'alarm': alarms.astype(bool)})
    hit_in_seizure = bool(np.any((case_df['y_true'] == 1) & case_df['alarm']))
    false_alarm_before = bool(np.any((case_df['time_rel_s'] < 0) & case_df['alarm']))
    first_alarm_rel_s = float(case_df.loc[case_df['alarm'], 'time_rel_s'].iloc[0]) if np.any(case_df['alarm']) else np.nan
    metrics = {'patient_id': patient_id, 'file_name': file_name, 'onset_epoch': onset_epoch, 'onset_time_s': onset_time_s, 'threshold': float(threshold), 'hit_in_seizure': hit_in_seizure, 'false_alarm_before': false_alarm_before, 'first_alarm_rel_s': first_alarm_rel_s}
    return (file_name, case_df, metrics)
