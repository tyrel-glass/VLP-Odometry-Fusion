# Data

This directory contains the Paper 2 data snapshot. Files are copied from the
original research repository and retain their original base filenames during
the first migration stage.

## Directory roles

- `raw/cnc`: original pipe-delimited CNC VLP fingerprint measurements.
- `raw/navigation_training`: original pipe-delimited robot/Vive collection
  runs.
- `processed/navigation_training`: processed comma-delimited datasets used by
  the RNN analysis.
- `processed/vlp_calibration`: consolidated VLP dataset retained from the
  original project.
- `raw/live_navigation`: final nominal live experiments, grouped by method.
- `raw/capture_experiments`: capture experiments, with EKF files excluded.

The manifest reports physical line counts minus the header as `data_rows`.
This is intentionally delimiter-independent and does not reinterpret or
rewrite any source CSV.
