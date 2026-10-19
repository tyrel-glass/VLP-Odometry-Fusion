# Paired live-recording estimator stress test

Fifty synthetic events per class are selected across nine nominal live recordings (three paths under three controllers). Each splice remains within one physical recording. The saved LSTM and primary position-only EKF replay the same event without resetting their internal state. Events are balanced across source runs and separated by at least 10 samples within each run. They reuse recorded motion and noise and are therefore dependent open-loop estimator tests, not repeated physical perturbations or controller-recovery trials.
