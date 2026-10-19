# Extracted RSS availability and scope (WP4; R2.5)

Navigation dataset 10 contains finite FFT-peak features for all four carriers on all 354 retained movement cycles, as well as archived ADC waveforms. Re-running the saved fingerprint model from the stored features reproduces the saved VLP positions to a maximum absolute coordinate difference of 1.657e-07 m.

The offline analysis applies common, single-carrier, and independent random multiplicative scaling to the extracted FFT-peak features before re-running the saved fingerprint model and LSTM. Capture normalization and FFT extraction precede the perturbations. Uniform scaling of a waveform cancels under peak normalization; changing one optical source can also alter normalization and the other features. These are therefore RSS-feature sensitivity tests, not physical source/receiver-gain or ambient-illumination experiments.

The receiver subtracts the sample mean, divides by the centered capture's positive peak, applies a Hann window, and extracts FFT peaks near 15, 17, 20, and 23 kHz. Mean removal and analog high-pass filtering attenuate DC and low-frequency ambient components while the front end remains linear. They do not establish immunity to increased shot noise, front-end saturation, or interference near a carrier. Controlled daylight, artificial-light, saturation, and near-carrier interference tests were not performed.
