# Archived Paper 2 RNN

The repository includes the trained model at `models/rnn/rnn.keras`. Its
serialized Keras configuration records the following final architecture:

- Input shape: 20 time steps by 8 features.
- First LSTM: 32 units, dropout 0.0, recurrent dropout 0.025.
- Layer normalization after the first LSTM.
- Second LSTM: 8 units, dropout 0.01, recurrent dropout 0.05.
- Linear two-value position output.
- Hyperbolic-tangent two-value heading output representing
  `[sin(theta), cos(theta)]`.
- Equal, unweighted mean-squared-error losses for the two outputs.
- Nadam optimizer with learning rate `0.0033727437257766724` and no learning
  rate scheduler.

These values were read directly from the archived model rather than inferred
from notebook output. The copied `rnn_config.py` has been aligned with this
configuration.

The eight input features, in order, are:

1. VLP x position.
2. VLP y position.
3. VLP-derived heading.
4. Change in VLP-derived heading.
5. Encoder heading change.
6. Encoder heading.
7. Encoder x position.
8. Encoder y position.
