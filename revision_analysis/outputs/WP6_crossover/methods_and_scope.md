# WP6 crossover-horizon method and scope

- Inputs: recorded live LSTM- and encoder-controlled square, hourglass, and random
  trials listed in `provenance.json`.
- Analysis windows: LSTM CSV rows 20--319 and encoder CSV rows 0--299, matching the
  300-sample windows used in the manuscript live-error comparison.
- Statistic: trailing 20-step median position error.
- Crossover: earliest one-based evaluation step for which the LSTM rolling median
  is strictly lower than the encoder rolling median at that step and every later
  step through step 300.
- Distance: cumulative Vive ground-truth displacement from the beginning of each
  analyzed trial window. LSTM and encoder distances are retained separately because
  they were separate closed-loop trials; the manuscript reports their range.
- Scope: descriptive of these three recorded paths. It is not a universal switch
  point and does not establish performance on another path, environment, or motion
  protocol.
