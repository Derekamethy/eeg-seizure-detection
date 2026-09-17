# Dataset

Obtain the CHB-MIT Scalp EEG Database from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) under its published terms. Raw EEG and annotation files are not redistributed here.

Pass the parent directory to `--data-root` with this layout:

```text
chb-mit/
  chb01/
    chb01-summary.txt
    chb01_01.edf
    ...
  ...
  chb10/
```

Each EDF must have an entry in its subject's summary, including zero-seizure recordings. The runner checks the requested cohort and uses strict 22-channel alignment; missing channels or invalid input stop the run. Both orientations of a bipolar channel are supported by reversing signal polarity where needed.

Keep recordings and generated feature caches outside version control. Cache hashes include configuration values; use `--rebuild-cache` after changing source recordings or annotations. The saved report's exact coverage cannot be confirmed without the original data and run records.
