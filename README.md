# EEG Seizure Detection from Multichannel Scalp EEG

Patient-specific seizure detection on **CHB-MIT scalp EEG**, using signal processing, engineered spectral and synchrony features, and a **Random Forest**. The reported study detected **53 of 55 seizures over 580.57 hours**, evaluated using event sensitivity, false alarms per hour, and detection delay.

[![Release QA](https://github.com/Derekamethy/eeg-seizure-detection/actions/workflows/release-qa.yml/badge.svg)](https://github.com/Derekamethy/eeg-seizure-detection/actions/workflows/release-qa.yml)

[Source code](src/eeg_seizure_detection/) · [Results](results/) · [Error analysis](results/error_analysis/) · [Academic report](docs/EE6019_Final_Report_PUBLIC_REDACTED.pdf)

## Overview

The study uses ten subjects (`chb01`–`chb10`). Each subject has a separate model, evaluated by holding out one seizure-containing recording and a disjoint subset of background recordings per outer fold. Inner training data determine feature selection; inner validation data determine the alarm threshold. Outer training then refits the classifier before held-out scoring.

This is retrospective, within-subject research. It does not establish performance on unseen patients or clinical readiness.

## Pipeline

```text
CHB-MIT EDF + seizure annotations
  → align 22 bipolar channels, including reversed-polarity handling
  → 0.5–50 Hz zero-phase Butterworth filter (order 4)
  → non-overlapping 2 s epochs; any seizure overlap gives a positive label
  → 440 FFT energy features + 3 inter-hemispheric correlations
  → current epoch + 3 previous epochs = 1,772 candidate inputs
  → inner-training Top-30 selection → patient-specific 500-tree RF (depth 12)
  → centred five-epoch median smoothing → validation threshold
  → retain runs of at least 3 epochs → event-level scoring
```

Temporal context is zero-padded at recording starts. The RF uses class weighting and training-only background subsampling. It needs no fitted feature scaling; the optional SVM comparator fits its scaler inside the training pipeline. There is no separate preictal class or notch-filter stage.

## Key results

| Reported final study metric | Value |
| --- | ---: |
| Subjects / EEG duration | 10 / 580.57 h |
| Detected seizure events | 53 / 55 |
| Mean subject event sensitivity | 0.9800 |
| Pooled event sensitivity | 0.9636 |
| Median subject false alarm rate | 0.2455 events/h |
| Pooled false alarm rate | 178 / 580.57 h = 0.3066 events/h |
| Mean of subject mean detected-event delays | 10.64 s |
| Pooled mean detected-event delay | 9.09 s |

These are the report's measurements, cross-checked against the saved [subject table](results/patient_metrics.csv). Macro and pooled averages answer different questions. The full-feature RF detected 52/55 events with 183 false alarms and a pooled delay of 6.69 s; Top-30 reduced false alarms and added one detection, at higher delay.

**Reproduction boundary:** the maintained evaluator now respects recording boundaries for smoothing, duration rules and event scoring, and applies smoothing consistently during validation and testing. These corrections have synthetic regression tests, but have not been rerun on the raw cohort. The saved study metrics are not measurements of the corrected evaluator.

## Engineering implementation

- EDF annotation parsing, channel alignment, filtering and configuration-keyed feature caches.
- Spectral/correlation feature extraction and past-epoch context stacking.
- Training-only feature ranking, seeded file splits and patient-specific RF training.
- Recording-aware event scoring, cohort summaries and explicit failures for incomplete inputs.
- Case-level error inspection, model serialization support, model-size and inference profiling.
- Executable JSON configuration, a command-line runner and synthetic regression tests in CI.

A true event is a contiguous positive-label run. Any alarm overlap detects it; delay is measured from its first positive epoch. False alarms count contiguous alarm fragments outside seizure labels, divided by **all evaluated hours**. A sustained alarm can overlap more than one seizure, and its non-seizure portions can count as false alarms. This is not one-to-one alarm matching.

## Repository structure

```text
src/eeg_seizure_detection/  Data, preprocessing, features, models, evaluation,
                           experiments, clinical inspection, deployment and I/O
configs/final_rf.json       Executable RF and feature configuration
scripts/                   Experiment runner and release checks
tests/                    Small synthetic and artifact-consistency tests
results/                   Reported metrics, subject table and error analysis
assets/                    Classical-model and feature-importance figures
docs/                      Privacy-redacted academic report
data/                      Dataset access and folder layout
.github/workflows/         Installation, release QA and unit tests
```

## Reproduce

Use Python 3.11 or later. Obtain CHB-MIT from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and follow the [dataset layout](data/README.md).

```bash
python -m pip install -e .
python scripts/check_release.py
python -m unittest discover -s tests -v
python scripts/run_final_rf.py --help
python scripts/run_final_rf.py --data-root /path/to/chb-mit --rebuild-cache
```

The last command requires the EDFs and annotations and performs the full experiment. It reads `configs/final_rf.json` and writes `outputs/reproduced_patient_summary.csv`. `--patients chb01` runs a single subject; `--config path/to/config.json` selects another configuration. Omit `--rebuild-cache` to reuse compatible caches. Rebuilding overwrites those caches. Missing annotations, channels, recordings or requested patient caches raise an error rather than silently reducing the cohort.

Optional model-family comparison support: `python -m pip install -e ".[benchmark]"` installs XGBoost. Random Forest reproduction does not require it.

## Results and outputs

- [Headline metrics](results/headline_metrics.csv), [subject metrics](results/patient_metrics.csv) and [feature-reduction comparison](results/feature_reduction_comparison.csv).
- [Classical-model benchmark](results/model_benchmark.csv) and [comparison figure](assets/final_multi_model_comparison.png): preliminary model-family comparison, with different RF settings from the final result. The figure labels median subject FAR as "Macro FAR".
- [Representative feature importance](assets/feature_importance_top20_chb01.png): one patient's ranking, not a universal feature set.
- [Error analysis](results/error_analysis/README.md): delayed detections, false alarms, accepted proxy augmentation and rejected refinements.
- [Academic report](docs/EE6019_Final_Report_PUBLIC_REDACTED.pdf): detailed methods, study results and deployment measurements. Its bibliography and future-work discussion provide broader research context.

The report profiles a representative 30-input, 500-tree model at approximately 2.99 MiB serialized size and 80.99 ms mean Python inference latency (Table 9). These are classifier-only software measurements, excluding EEG preprocessing and feature extraction; they are not target-hardware measurements. A compact C-export comparison reduced size from about 3,642.3 KB to 719.2 KB, with lower sensitivity (0.955 to 0.930) and higher FAR (0.2617/h to 0.4234/h).

## Limitations

- Zero-phase filtering, centred smoothing and retrospective duration filtering are noncausal. Reported delays use epoch starts and exclude acquisition/confirmation latency.
- Binary epoch labels quantize onset and can merge nearby annotated events; results depend on that event definition.
- Repeated configuration selection on the same cohort can produce optimistic estimates despite training/validation separation within folds.
- The error-analysis study covers 54 events over 564.56 h and must be compared against its own baseline. Its case-derived refinements need independent validation.
- Raw EEG, prediction traces and fitted models are not distributed. Installation and synthetic tests validate software behaviour, not full-data reproduction of the saved results. The cause of the study/error-analysis coverage difference cannot be fully resolved without those inputs.

## Dataset, license and citation

CHB-MIT data are distributed separately by [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). Source code is under the [MIT License](LICENSE); dataset and third-party material retain their own terms, as described in [NOTICE.md](NOTICE.md). Cite the software using [CITATION.cff](CITATION.cff).

Yangdeyi Yang · UCC EE6019 Research Project
