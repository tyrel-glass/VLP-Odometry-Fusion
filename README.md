# VLP–odometry navigation: Thesis Paper 2 reproducibility repository

This repository is a standalone snapshot of the data, trained models, and
analysis code used for Paper 2 of Tyrel Glass's thesis. It covers visible-light
positioning (VLP), encoder odometry, recurrent neural-network (RNN) fusion,
nominal live-navigation experiments, and controlled capture experiments.
It is intentionally an analysis repository: hardware control, experimental
data acquisition, firmware, and live robot-driving code are excluded.

Extended Kalman filter (EKF) experimental data is deliberately excluded. The
EKF implementation used during the original experiments contained errors and
is not part of the claims reproduced here.

## Repository status

The migration began on 21 September 2026. The source data and trained models
have been copied without modifying the original `robot-vlp` repository. The
notebooks and Python modules are preserved as migration sources and are now
being converted into a clean, repository-relative workflow. See
[`docs/migration_status.md`](docs/migration_status.md) for the remaining work.

## Included data

- Two CNC VLP fingerprint datasets.
- Ten raw robot-navigation data-collection runs.
- Ten processed navigation runs used for RNN development and evaluation.
- Final RNN, VLP-only, and odometry-only live runs for square, hourglass, and
  random paths.
- Non-EKF capture experiments involving translation, rotation, or both.
- The saved RNN and VLP models used by the analysis.

The zero-based source files `exp1_0` through `exp1_9` correspond to thesis
Datasets 1 through 10. Dataset roles are documented in
[`docs/dataset_splits.md`](docs/dataset_splits.md).

## Data integrity and provenance

[`data/manifest.csv`](data/manifest.csv) records the source-repository path,
size, row count, SHA-256 digest, and experiment group for each CSV file.
[`checksums.sha256`](checksums.sha256) covers all migrated data and model
artifacts.

All CSV files are currently stored directly in Git, as requested. Git LFS is
not configured.

## Preliminary setup

The project targets Python 3.10. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebooks,tuning]'
```

The dependency versions still need to be frozen against the environment used
to produce the thesis results before the repository is released.

## Layout

```text
data/raw/cnc/                         CNC VLP measurements
data/raw/navigation_training/         ten original robot runs
data/processed/navigation_training/   ten processed RNN datasets
data/raw/live_navigation/             final RNN, VLP, and odometry runs
data/raw/capture_experiments/          non-EKF capture runs
models/                               serialized RNN and VLP models
src/paper2_navigation/                analysis and modelling modules
notebooks/                            curated source notebooks
scripts/                              reproducibility and validation tools
docs/                                 experiment and migration documentation
results/                              generated tables and figures
```

## Rebuilding the manifest

```bash
python scripts/build_manifest.py
```

## Licensing

No reuse license has yet been selected. A code license, data license, and
third-party attribution review must be completed before public release.
