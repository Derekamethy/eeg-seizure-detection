"""Lightweight checks for the files shipped in the public repository."""
from pathlib import Path
import ast
import csv
import json
import re
import subprocess
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "LICENSE", "NOTICE.md", "CITATION.cff", "pyproject.toml",
    "configs/final_rf.json", "scripts/run_final_rf.py", "tests/test_core_behaviour.py",
    "src/eeg_seizure_detection/config.py", "src/eeg_seizure_detection/data.py",
    "src/eeg_seizure_detection/preprocessing.py", "src/eeg_seizure_detection/features.py",
    "src/eeg_seizure_detection/models.py", "src/eeg_seizure_detection/evaluation.py",
    "results/headline_metrics.csv", "results/patient_metrics.csv",
    "results/error_analysis/README.md", ".github/workflows/release-qa.yml",
    "docs/EE6019_Final_Report_PUBLIC_REDACTED.pdf",
]
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".toml", ".txt", ".cff", ".ipynb", ".yml", ".yaml"}
OBSOLETE = ("clinical_error_analysis/", "notebooks/reference/", "extensions/cnn/",
            "legacy_core", "PROJECT_AUDIT", "Historical Reference", "cell-level lineage")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def release_files():
    # Respect Git ignores: never descend into datasets, outputs or virtual environments.
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True)
    return sorted({ROOT / name for name in result.stdout.decode("utf-8").split("\0")
                   if name and (ROOT / name).is_file()})


def check_links(path, text, errors):
    for target in LINK.findall(text):
        target = target.strip().split(' "', 1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("//"):
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(ROOT.resolve()):
            errors.append(f"Link escapes repository: {path.relative_to(ROOT)} -> {target}")
        elif not resolved.exists():
            errors.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")


def main():
    errors = []
    files = release_files()
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required file: {name}")
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Unexpected file larger than 10 MiB: {rel}")
        if path.suffix.lower() in {".edf", ".joblib", ".pkl", ".npz", ".npy", ".pem", ".key"} or path.name == ".env":
            errors.append(f"Data, model or private file in release: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix == ".py":
                ast.parse(text, filename=rel)
            if path.suffix == ".json":
                json.loads(text)
            if path.suffix == ".toml":
                tomllib.loads(text)
            if path.suffix == ".csv":
                rows = list(csv.reader(text.splitlines()))
                if not rows or len(set(rows[0])) != len(rows[0]) or any(len(r) != len(rows[0]) for r in rows[1:]):
                    errors.append(f"Malformed CSV: {rel}")
            if path.suffix == ".md":
                check_links(path, text, errors)
            if path.suffix == ".ipynb":
                for cell in json.loads(text)["cells"]:
                    if cell.get("outputs") or cell.get("execution_count") is not None:
                        errors.append(f"Saved notebook execution state: {rel}")
                    source = "".join(cell.get("source", []))
                    if cell["cell_type"] == "markdown":
                        check_links(path, source, errors)
                    elif cell["cell_type"] == "code":
                        ast.parse(source)
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/](?!/)", text):
                errors.append(f"Absolute Windows path: {rel}")
            if path != Path(__file__).resolve() and re.search(r"-----BEGIN .*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}", text):
                errors.append(f"Possible secret: {rel}")
            if path != Path(__file__).resolve():
                if any(term.lower() in text.lower() for term in OBSOLETE) or re.search(r"\bcnn\b", text, re.I):
                    errors.append(f"Obsolete content or reference: {rel}")
        except (ValueError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Invalid {rel}: {exc}")
    if errors:
        print("RELEASE QA FAILED\n" + "\n".join(f"- {e}" for e in errors))
        return 1
    print(f"RELEASE QA PASSED ({len(files)} files; syntax, links, tables, notebook state and release hygiene)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
