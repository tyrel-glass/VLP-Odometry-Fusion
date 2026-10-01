# WP1 EKF comparison

The primary filter uses VLP position only. Secondary filters use one-step and 7-interval line-fit VLP heading. Process and full measurement covariances were estimated on manuscript datasets 1--8; dataset 9 selected only the short window; dataset 10 was held out. Both EKF and LSTM were initialized from the first two usable VLP fixes, without Vive state. Results use common samples and circular heading error. The EKF comparison is offline; no closed-loop EKF run was recorded.

The comparison uses the first 300 post-window samples from dataset 10 (raw zero-based indices 19--318). The lowest all-sample median position error was produced by Paper LSTM (deployable initialization); the lowest median heading error was produced by EKF: short-window VLP heading (7 intervals). Refer to held_out_metrics.csv for values and segment-specific comparisons.
