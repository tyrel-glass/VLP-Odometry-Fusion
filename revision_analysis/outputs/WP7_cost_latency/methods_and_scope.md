# WP7 training-cost and latency method and scope

## Training evidence

- Datasets 1--8 contain 12,150 processed rows and produce
  11,958 run-bounded 25-sample, stride-1 windows.
- The saved final fit ran for 222 epochs at batch size 32. Summing
  Keras' integer per-epoch durations gives 447 s
  (7.45 min); the notebook does not retain exact
  cell wall-clock metadata.
- The notebook explicitly disabled GPU visibility, so training was CPU-only. The
  historical processor and RAM were not logged and are not inferred from the
  current computer.
- The four tuner directories contain 605 completed trials.
  Creation-to-final-write intervals total 21.35 h. This
  is an approximate active-trial total, not a recorded end-to-end search duration.
- Raw datasets 1--8 contain 15,740 logged cycles; 12,150
  have VLP data and were retained. Explicit delays in the archived loop imply an
  approximate 11.0 h protocol-delay subtotal. Robot
  motion, command/network round trips, retries, and scheduling, so total operating
  time was longer and cannot be recovered exactly.

## Latency evidence

- One 512-sample acquisition at 50 kHz is exactly 10.24 ms. The navigation loop
  requested three captures per movement cycle.
- UART values are calculated from 20,448 complete 512-sample frames in the two CNC
  files, decimal-text payload size, 460800 baud, and 8N1 framing. They are not
  measured time intervals.
- TCP chunk-pacing values come from the archived firmware's 10 ms delay per 50-byte
  chunk. They do not include actual Wi-Fi transfer time or host scheduling.
- FFT/peak extraction, fingerprint inference, and LSTM inference were newly measured
  with `time.perf_counter_ns` on MacBook Air,
  Apple M3, 24 GB,
  using Python 3.10.14 and TensorFlow 2.16.2 on
  `/CPU:0`. These warmed, serial component timings are illustrative current-host
  measurements, not historical live-system timestamps.
- Navigation and live-result CSVs have no timestamp columns. Therefore Wi-Fi RTT,
  command return/handling, and historical end-to-end latency median/P90 are
  unavailable. No synthetic total is reported.
- The robot stopped before sensing. Latency therefore reduced trial throughput in
  the evaluated protocol; under continuous motion it would instead produce a
  velocity-dependent spatial lag unless compensated.
