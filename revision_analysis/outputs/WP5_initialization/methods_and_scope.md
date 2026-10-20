# Repeated live-segment initialization sensitivity

The nine nominal live recordings are divided into 50 non-overlapping 70-sample blocks. For each block, the first VLP fix and the following fix initialize a two-fix VLP heading. Both conditions begin at the second fix: the known-pose condition uses Vive position and heading at that same instant, whereas the deployable condition uses the second VLP position and heading derived from the two fixes. Encoder absolute-pose features are then re-integrated from each initial state using identical recorded increments. Fifty paired predictions are evaluated, starting at the end of the first complete 20-sample LSTM window. Prediction index 1 denotes the estimate at the end of that first window. The saved LSTM, normalization, feature order, and weights are unchanged. Segments do not overlap, but segments from the same physical recording remain dependent; paired-difference confidence intervals therefore resample complete source runs. This is an offline estimator replay and does not reproduce controller reactions to alternative initialization.

The reviewer-response table reports cross-segment median and P90 errors at prediction
steps 1, 15, 30, and 45. Complete 50-step values remain in
`segment_error_traces.csv`.
