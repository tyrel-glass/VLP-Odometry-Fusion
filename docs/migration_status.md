# Migration status

## Completed

- Created a repository separate from `robot-vlp`.
- Copied both raw CNC VLP datasets.
- Copied all ten raw and processed navigation runs.
- Copied the final RNN, VLP-only, and odometry-only live runs.
- Copied non-EKF capture files.
- Copied the saved RNN and VLP models.
- Copied the four principal source notebooks.
- Retained the relevant non-EKF processing, RNN, statistics, and plotting
  modules.
- Removed serial, Vive acquisition, firmware, robot-control, and path-driving
  code; pure transformations of recorded measurements were retained.
- Replaced the old package namespace in the copied source tree.
- Added a repository-relative configuration module.
- Added integrity and provenance tooling.
- Executed all four notebooks from a clean kernel and validated the remaining
  command-line analysis entry points.
- Removed unused project-template and legacy simulation plotting modules that
  depended on datasets and model names outside this repository.

## Still required before public release

- Confirm which duplicate capture files correspond to the final thesis plots.
- Remove duplicate or superseded capture files after preserving their hashes.
- Finish adapting notebook data paths to the new directory structure.
- Separate reusable analysis from exploratory notebook cells.
- Remove any remaining obsolete analysis code paths.
- Freeze exact dependency versions, especially TensorFlow and Keras.
- Run every analysis from a fresh environment.
- Compare regenerated tables and figures against the final thesis.
- Write a complete column-level data dictionary, including units and frames.
- Add code and data licenses, citation metadata, and third-party attribution.

## EKF exclusion

No file whose name contains `ekf` was migrated. The copied navigation-dataset
notebook is being stripped of old EKF-only analysis cells before it is treated
as executable documentation. EKF results must not be reintroduced into this
repository without a separate, explicitly validated study.
