# Fingerprint-regressor comparison (WP3; R1.6)

All methods use the paper's four FFT-peak features, the same 1,517-point CNC training grid, and the independent 528-point perturbed-grid validation set. The saved feedforward model is evaluated without retraining. Inverse-distance-weighted kNN selects k=7 by five-fold training-only cross-validation over k=1--15. Separate x/y Gaussian-process regressors use a constant-times-anisotropic-RBF plus white-noise kernel; hyperparameters are fitted by marginal likelihood using training data only.

Memory is the serialized estimator plus required preprocessing object. Keras serialization contains framework metadata as well as weights, so it is an artifact-size comparison rather than bare parameter memory. Query times are CPU measurements for a 528-sample batch divided by batch size; they are host measurements, not embedded-device benchmarks.
