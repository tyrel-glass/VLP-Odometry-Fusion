# WP6 crossover-horizon method and scope

- Inputs: recorded live LSTM- and encoder-controlled square, hourglass, and random
  trials, plus the common held-out offline LSTM and encoder-odometry error traces,
  as listed in `provenance.json`.
- Analysis windows: LSTM CSV rows 20--319 and encoder CSV rows 0--299, matching the
  300-sample windows used in the manuscript live-error comparison.
- Statistics: separate trailing 20-step medians for position error and circular
  absolute heading error.
- Crossover: earliest one-based evaluation step for which the LSTM rolling median
  is strictly lower than the encoder rolling median at that step and every later
  step through step 300.
- Distance: one-based crossover step multiplied by the measured mean robot step
  length of 0.1118 m.
- Scope: descriptive of these recordings and each reported metric. It is not a universal switch
  point and does not establish performance on another path, environment, or motion
  protocol.
