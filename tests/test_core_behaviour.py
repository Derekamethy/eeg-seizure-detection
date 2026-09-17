import unittest
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import mne

import numpy as np

from eeg_seizure_detection.evaluation import (
    compute_event_metrics,
    count_events,
    extract_binary_runs,
    choose_threshold_from_validation,
    smooth_recordings,
    threshold_recordings,
    split_inner_validation_files,
    evaluate_patient_loso,
    evaluate_many_patients,
    build_event_detection_matrix,
)
from eeg_seizure_detection.features import temporal_stack_features, extract_base_features_for_file, build_stacked_feature_names
from eeg_seizure_detection.config import make_final_rf_config, load_config
from eeg_seizure_detection.io_utils import config_to_hash
from eeg_seizure_detection.data import parse_summary_to_dict, build_patient_cache
from eeg_seizure_detection.models import train_patient_bundle, make_model, select_top_k_features_tree
from eeg_seizure_detection.preprocessing import build_epoch_labels, align_channels, bandpass_filter_multich
from eeg_seizure_detection.deployment import profile_rf_model, benchmark_single_epoch_latency

ROOT = Path(__file__).resolve().parents[1]
from eeg_seizure_detection.evaluation import apply_duration_constraint


class CoreBehaviourTests(unittest.TestCase):
    def test_duration_constraint_removes_short_runs(self):
        preds = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int8)
        cleaned = apply_duration_constraint(preds, min_epochs=3)
        np.testing.assert_array_equal(
            cleaned,
            np.array([0, 0, 0, 0, 1, 1, 1, 0], dtype=np.int8),
        )

    def test_temporal_stack_preserves_causal_history_order(self):
        x = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.int8)
        stacked, y_out = temporal_stack_features(x, y, history_epochs=1)
        expected = np.array(
            [[0, 0, 1, 10], [1, 10, 2, 20], [2, 20, 3, 30]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(stacked, expected)
        np.testing.assert_array_equal(y_out, y)

    def test_event_metrics_count_events_false_alarms_and_delay(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)
        self.assertEqual(count_events(y_true), 2)
        starts, ends = extract_binary_runs(y_true)
        np.testing.assert_array_equal(starts, np.array([1, 5]))
        np.testing.assert_array_equal(ends, np.array([3, 7]))

        metrics = compute_event_metrics(y_true, y_pred, epoch_len_s=2)
        self.assertEqual(metrics["events"], 2.0)
        self.assertEqual(metrics["detected_events"], 2.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["false_alarm_events"], 1.0)
        self.assertAlmostEqual(metrics["mean_delay_s"], 1.0)
        self.assertAlmostEqual(metrics["median_delay_s"], 1.0)

    def test_config_file_matches_final_defaults_and_rejects_unknown_fields(self):
        cfg = load_config(ROOT / 'configs/final_rf.json', 'data/chb-mit')
        self.assertEqual(config_to_hash(cfg), config_to_hash(make_final_rf_config('data/chb-mit')))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{"eval": {"typo": 2}}')
            with self.assertRaises(TypeError):
                load_config(path, tmp)
        with self.assertRaises(ValueError):
            load_config(ROOT / 'configs/final_rf.json', '.', [])

    def test_config_mutation_invalidates_hash_and_feature_names(self):
        cfg = make_final_rf_config('.')
        before = config_to_hash(cfg)
        self.assertEqual(len(build_stacked_feature_names(cfg)), 1772)
        cfg.feature.history_epochs = 1
        self.assertNotEqual(config_to_hash(cfg), before)
        self.assertEqual(len(build_stacked_feature_names(cfg)), 886)
        cfg.feature.band_edges_hz = (0, 10, 20)
        self.assertEqual(len(build_stacked_feature_names(cfg)), 94)

    def test_synthetic_features_shape_finite_and_spectral_location(self):
        cfg = make_final_rf_config('.')
        signal = np.sin(2 * np.pi * 9 * np.arange(1025) / 256)
        x, indices = extract_base_features_for_file(np.tile(signal, (22, 1)), 256, cfg)
        self.assertEqual(x.shape, (2, 443))
        np.testing.assert_array_equal(indices, [0, 1])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(np.argmax(x[0, :20]), 4)
        np.testing.assert_allclose(x[:, -3:], 1)
        zero, _ = extract_base_features_for_file(np.zeros((22, 512)), 256, cfg)
        np.testing.assert_array_equal(zero, np.zeros((1, 443)))
        with self.assertRaises(ValueError):
            temporal_stack_features(x, np.zeros(2), -1)

    def test_channel_reversal_and_missing_channel_policy(self):
        raw = mne.io.RawArray(np.array([[1., 2., 3.]]), mne.create_info(['F7-FP1'], 256, 'eeg'), verbose=False)
        aligned, info = align_channels(raw, ['FP1-F7'])
        np.testing.assert_array_equal(aligned, [[-1, -2, -3]])
        self.assertEqual(info['reversed_channels'], ['FP1-F7'])
        with self.assertRaises(ValueError):
            align_channels(raw, ['F3-C3'])

    def test_filter_attenuates_out_of_band_signal(self):
        time = np.arange(4096) / 256
        data = np.sin(2*np.pi*10*time) + np.sin(2*np.pi*90*time)
        filtered = bandpass_filter_multich(data[None], 256, .5, 50)[0]
        residual = filtered[512:-512] - np.sin(2*np.pi*10*time[512:-512])
        self.assertLess(np.std(residual), .03)

    def test_annotation_parser_numbered_events_and_missing_or_invalid_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.txt'
            with self.assertRaises(FileNotFoundError):
                parse_summary_to_dict(path)
            text = ('File Name: chb01_01+.edf\nNumber of Seizures in File: 2\n'
                    'Seizure 1 Start Time: 2 seconds\nSeizure 1 End Time: 4 seconds\n'
                    'Seizure 2 Start Time: 7 seconds\nSeizure 2 End Time: 9 seconds\n')
            path.write_text(text)
            self.assertEqual(parse_summary_to_dict(path), {'chb01_01+.edf': [(2, 4), (7, 9)]})
            path.write_text(text.replace('File: 2', 'File: 3'))
            with self.assertRaises(ValueError):
                parse_summary_to_dict(path)
        np.testing.assert_array_equal(build_epoch_labels(5, 2, [(2, 4), (7, 9)]), [0, 1, 0, 1, 1])

    def test_empty_patient_directory_is_not_a_successful_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / 'chb01'
            patient.mkdir()
            (patient / 'chb01-summary.txt').write_text('')
            with self.assertRaises(ValueError):
                build_patient_cache(make_final_rf_config(tmp, ['chb01']), 'chb01')

    def test_event_boundaries_hours_and_empty_cases(self):
        y = np.ones(6, dtype=np.int8)
        metrics = compute_event_metrics(y, y, 2, [3, 3])
        self.assertEqual(metrics['events'], 2)
        self.assertEqual(metrics['detected_events'], 2)
        metrics = compute_event_metrics(np.zeros(6), y, 2, [3, 3])
        self.assertEqual(metrics['false_alarm_events'], 2)
        self.assertAlmostEqual(metrics['far_per_hour'], 600)
        self.assertTrue(np.isnan(metrics['sensitivity']))
        self.assertTrue(np.isnan(compute_event_metrics([], [], 2)['far_per_hour']))
        with self.assertRaises(ValueError):
            compute_event_metrics(y, y, 2, [5])
        with self.assertRaises(ValueError):
            compute_event_metrics(y, y[:-1], 2)

    def test_delay_aggregation_weights_detected_events(self):
        metrics = compute_event_metrics([1,1,0,1,1,0,1,1], [0,1,0,1,0,0,0,0], 2, [3,5])
        self.assertEqual(metrics['events'], 3)
        self.assertEqual(metrics['detected_events'], 2)
        self.assertEqual(metrics['mean_delay_s'], 1)
        self.assertEqual(metrics['median_delay_s'], 1)

    def test_postprocessing_never_crosses_recording_boundaries(self):
        scores = np.array([0.,0.,0.,1.,1.,1.,0.,0.])
        smooth = smooth_recordings(scores, [4, 4])
        np.testing.assert_array_equal(smooth, [0,0,0,0,0,0,0,0])
        pred = threshold_recordings(np.ones(4), .5, 3, [2,2])
        np.testing.assert_array_equal(pred, np.zeros(4))

    def test_threshold_selection_uses_smoothed_validation_scores(self):
        cfg = make_final_rf_config('.')
        cfg.eval.min_duration_epochs = 1
        cfg.eval.threshold_grid = (.5, .8)
        y = np.array([0,0,1,1,1,0,0])
        # The isolated .9 spike disappears under the five-epoch median.
        score = np.array([0.,.6,.6,.9,.6,.6,0.])
        result = choose_threshold_from_validation(y, score, cfg)
        self.assertEqual(result['threshold'], .5)
        self.assertEqual(result['sensitivity'], 1)

    def test_inner_split_is_disjoint_complete_and_seeded(self):
        seizure, background = ['s1','s2','s3'], ['b1','b2','b3','b4']
        split = split_inner_validation_files(seizure, background, 42)
        self.assertEqual(split, split_inner_validation_files(seizure, background, 42))
        train = split[0] + split[1]
        val = split[2] + split[3]
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(seizure + background))

    def test_outer_test_files_never_enter_fitting_or_selection(self):
        cfg = make_final_rf_config('.', ['chb01'])
        cfg.eval.min_duration_epochs = 1
        files = {}
        for i in range(6):
            y = np.array([0,0,1,1,1,0,0,0]) if i < 3 else np.zeros(8, dtype=int)
            files[str(i)] = {'X': np.column_stack([np.full(8, i), y, y]), 'y': y, 'has_seizure': i < 3}
        payload = {'meta': {'patient_id': 'chb01'}, 'files': files}
        calls = []
        selected = []
        class FakeModel:
            def fit(self, x, y):
                self.fit_ids = set(x[:, 0]); calls.append(self); return self
            def predict_proba(self, x):
                self.predicted_ids = set(x[:, 0])
                return np.column_stack([1-x[:,1], x[:,1]])
        def selector(x, y, **kwargs):
            selected.append(set(x[:,0])); return np.array([0,1])
        with patch('eeg_seizure_detection.evaluation.make_model', side_effect=lambda *a, **k: FakeModel()), patch('eeg_seizure_detection.evaluation.select_top_k_features_tree', side_effect=selector):
            result = evaluate_patient_loso(payload, cfg, model_name='random_forest', top_k=2)
        for fold in range(3):
            inner, outer = calls[2*fold:2*fold+2]
            self.assertEqual(selected[fold], inner.fit_ids)
            self.assertFalse(inner.fit_ids & inner.predicted_ids)
            self.assertFalse(outer.fit_ids & outer.predicted_ids)
            self.assertFalse(selected[fold] & outer.predicted_ids)
        self.assertEqual(result['summary']['True_Seizures'], 3)
        self.assertEqual(len(result['recording_lengths']), 6)
        matrix = build_event_detection_matrix({'patient_outputs': {'chb01': result}}, cfg, 'RF')
        self.assertEqual(matrix.iloc[0]['TP_events'], 3)

    def test_real_rf_evaluation_is_deterministic_on_small_cohort(self):
        cfg = make_final_rf_config('.', ['chb01'])
        cfg.eval.rf_n_estimators = cfg.eval.selector_n_estimators = 3
        cfg.eval.rf_n_jobs = 1
        cfg.eval.threshold_grid = (.3, .5, .7)
        files = {}
        for i in range(6):
            y = np.array([0]*5 + [1]*5 + [0]*5) if i < 3 else np.zeros(15, dtype=int)
            files[str(i)] = {'X': np.column_stack([y, 1-y, np.zeros(15)]).astype(np.float32),
                             'y': y, 'has_seizure': i < 3}
        payload = {'meta': {'patient_id': 'chb01'}, 'files': files}
        cache = {}
        first = evaluate_patient_loso(payload, cfg, top_k=2, fold_topk_cache=cache)
        second = evaluate_patient_loso(payload, cfg, top_k=2, fold_topk_cache=cache)
        np.testing.assert_array_equal(first['y_pred'], second['y_pred'])
        self.assertEqual(second['topk_cache_hits'], 3)
        self.assertEqual(first['summary']['Detected_Seizures'], 3)
        self.assertEqual(first['summary']['False_Alarm_Events'], 0)
        cfg.eval.min_neg_samples += 1
        third = evaluate_patient_loso(payload, cfg, top_k=2, fold_topk_cache=cache)
        self.assertEqual(third['topk_cache_hits'], 0)

    def test_missing_requested_patient_raises(self):
        cfg = make_final_rf_config('.', ['chb01'])
        with self.assertRaises(ValueError):
            evaluate_many_patients({}, cfg)

    def test_training_bundle_roundtrip_and_profile(self):
        cfg = make_final_rf_config('.', ['chb01'])
        cfg.eval.rf_n_estimators = cfg.eval.selector_n_estimators = 3
        cfg.eval.rf_n_jobs = 1
        rng = np.random.default_rng(42)
        x = rng.normal(size=(20, 1772)).astype(np.float32)
        y = np.tile([0,1], 10)
        payload = {'files': {'sample': {'X': x, 'y': y}}}
        bundle = train_patient_bundle('chb01', payload, cfg, .5)
        self.assertEqual(bundle['input_dim'], 30)
        buffer = io.BytesIO()
        joblib.dump(bundle, buffer); buffer.seek(0)
        restored = joblib.load(buffer)
        light = x[:, bundle['top_k_indices']]
        np.testing.assert_array_equal(bundle['model'].predict_proba(light), restored['model'].predict_proba(light))
        self.assertEqual(profile_rf_model(bundle['model'], 30)['n_trees'], 3)
        self.assertEqual(make_model('random_forest', cfg).random_state, 42)
        with self.assertRaises(ValueError):
            benchmark_single_epoch_latency(bundle['model'], light[:0])
        with self.assertRaises(ValueError):
            select_top_k_features_tree(x, y, 0, cfg)

    def test_reported_results_reconcile_subject_macro_and_pooled_values(self):
        with (ROOT / 'results/patient_metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        with (ROOT / 'results/headline_metrics.csv').open() as f:
            headline = {r['metric']: float(r['value']) for r in csv.DictReader(f)}
        hours = sum(float(r['hours']) for r in rows)
        detected = [round(float(r['total_events']) * float(r['sensitivity'])) for r in rows]
        delays = [float(r['mean_detected_event_delay_s']) for r in rows]
        self.assertAlmostEqual(hours, headline['eeg_hours'], places=4)
        self.assertEqual(sum(detected), headline['pooled_detected_events'])
        self.assertEqual(sum(float(r['total_events']) for r in rows), headline['pooled_total_events'])
        self.assertAlmostEqual(np.mean([float(r['sensitivity']) for r in rows]), headline['macro_event_sensitivity'])
        self.assertAlmostEqual(np.median([float(r['far_per_hour']) for r in rows]), headline['median_subject_far_per_hour'], places=4)
        self.assertAlmostEqual(np.mean(delays), headline['macro_mean_detected_event_delay_s'], places=2)
        self.assertAlmostEqual(np.average(delays, weights=detected), headline['pooled_mean_detected_event_delay_s'], places=2)
        self.assertAlmostEqual(headline['pooled_false_alarm_events'] / hours, headline['pooled_far_per_hour'], places=4)

    def test_cli_help(self):
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/run_final_rf.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--config', run.stdout)


if __name__ == "__main__":
    unittest.main()
