# Final hyperparameters encoded by the archived Paper 2 model.
_DEFAULT = {
    "best_architecture": {
        "num_layers": 2,
        "recurrent_units": [32, 8],
    },
    "sequence_length": {
        "input_length": 25,
        "use_length": 20,
    },
    "regularization_defaults": {
        "dropout": [0.0, 0.01],
        "recurrent_dropout": [0.025, 0.05],
        "batch_norm": [False, False],
        "layer_norm": [True, False],
    },
    "optimization_defaults": {
        "lr": 0.0033727437257766724,
        "scheduler": "none",
        "optimizer": "nadam",
    },
}

GLOBAL_CONFIG = _DEFAULT
