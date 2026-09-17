"""Re-run the canonical patient-specific RF when CHB-MIT data is available."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_seizure_detection.config import load_config
from eeg_seizure_detection.data import build_all_caches, load_all_caches
from eeg_seizure_detection.evaluation import evaluate_many_patients


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="CHB-MIT dataset root")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "final_rf.json")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "reproduced_patient_summary.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.data_root, args.patients)
    if args.rebuild_cache:
        build_all_caches(cfg, overwrite=True)
    caches = load_all_caches(cfg)
    if not caches:
        raise RuntimeError(
            "No compatible feature caches found. Re-run with --rebuild-cache."
        )

    result = evaluate_many_patients(
        caches=caches,
        cfg=cfg,
        model_name="random_forest",
        top_k=cfg.eval.top_k_features,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result["summary_df"].to_csv(output, index=False)
    print(result["summary_df"].to_string(index=False))
    print(f"\nSaved patient summary to: {output}")


if __name__ == "__main__":
    main()
