# Raw RSS availability and scope (WP4; R2.5)

Held-out navigation dataset 10 contains finite FFT-peak amplitudes for all four carriers on all 354 retained movement cycles, as well as archived ADC waveforms. Re-running the saved fingerprint model from the stored four peaks reproduces the saved VLP positions to a maximum absolute coordinate difference of 1.657e-07 m.

The offline analysis applies common receiver/source gain changes, one-carrier gain changes, and independent random multiplicative changes to the recorded FFT peaks before re-running the saved fingerprint model and LSTM. These perturbations characterize modulated-source/receiver-gain sensitivity only. They are not measurements of ambient illumination.

The receiver subtracts the sample mean, applies a Hann window, and extracts peaks near 15, 17, 20, and 23 kHz. This rejects DC and attenuates low-frequency ambient variation while the analog front end remains linear. It does not establish immunity to increased shot noise, front-end saturation, or interference near a carrier. Controlled daylight, artificial-light, saturation, and near-carrier interference tests were not performed.
