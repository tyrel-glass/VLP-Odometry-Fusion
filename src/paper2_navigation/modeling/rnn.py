
# model_builders.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Model
from paper2_navigation.modeling.rnn_config import GLOBAL_CONFIG



global_norm = layers.Normalization(axis=-1, name="feature_norm")


# Helper Function
def slice_last_n_timesteps(n):
    return layers.Lambda(lambda t: t[:, -n:, :], name=f"slice_last_{n}")

# Stage 1: Architecture Tuning (unchanged, but using config for sequence length)
def build_architecture_model(hp):

    seq_len = GLOBAL_CONFIG["sequence_length"]["use_length"]
    inputs = keras.layers.Input(shape=(GLOBAL_CONFIG["sequence_length"]["input_length"], 8), name="input")

    x = global_norm(inputs)
    x = slice_last_n_timesteps(seq_len)(x)

    fixed_dropout = 0.0
    fixed_lr = 1e-3

    num_layers = hp.Int("num_layers", 1, 4)
    for i in range(num_layers):
        units = hp.Choice(f"recurrent_units_{i}", values=[8,16,32,64,128])
        return_sequences = (i < num_layers - 1)
        x = layers.LSTM(units,
                       return_sequences=return_sequences,
                       dropout=fixed_dropout,
                       recurrent_dropout=fixed_dropout,
                       name=f'lstm_{i}')(x)

    loc_output = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = layers.Dense(2, activation='tanh', name='angle_output')(x)

    model = Model(inputs=inputs, outputs=[loc_output, angle_output])
    optimizer = keras.optimizers.Adam(learning_rate=fixed_lr)
    model.compile(optimizer=optimizer,
                  loss={'loc_output':'mse','angle_output':'mse'})
    return model

# Stage 2: Regularization Tuning
def build_regularization_model(hp):
    cfg = GLOBAL_CONFIG
    arch = cfg['best_architecture']
    seq_len = cfg['sequence_length']['use_length']
    inputs = keras.layers.Input(shape=(cfg['sequence_length']['input_length'], 8), name="input")

    x = global_norm(inputs)
    x = slice_last_n_timesteps(seq_len)(x)

    for i in range(arch['num_layers']):
        units = arch['recurrent_units'][i]
        dr = hp.Choice(
            f"dropout_{i}",
            values=[0.0, 0.01, 0.025, 0.05]
        )
        rdr = hp.Choice(
            f"recurrent_dropout_{i}",
            values=[0.0, 0.01, 0.025, 0.05]
        )
        return_sequences = (i < arch['num_layers'] - 1)

        x = layers.LSTM(units,
                       return_sequences=return_sequences,
                       dropout=dr,
                       recurrent_dropout=rdr,
                       name=f'lstm_{i}')(x)
        if hp.Boolean(f"batch_norm_{i}"):
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        if hp.Boolean(f"layer_norm_{i}"):
            x = layers.LayerNormalization(name=f'layer_norm_{i}')(x)

    loc_output = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = layers.Dense(2, activation='tanh', name='angle_output')(x)

    model = Model(inputs=inputs, outputs=[loc_output, angle_output])
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer,
                  loss={'loc_output':'mse','angle_output':'mse'})
    return model

def build_optimization_model(hp):
    """
    Stage 3: Optimization Tuning
    - LR: 1e-4 → 1e-2 (log scale)
    - Scheduler: none / cosine / exponential
      • cosine: decay_steps ∈ [2k,20k]
      • exponential: decay_steps ∈ [1k,10k], decay_rate ∈ [0.8,0.99]
    - Optimizer: Adam / RMSprop / SGD / Nadam
    """


    cfg = GLOBAL_CONFIG
    arch = cfg['best_architecture']
    seq_len = cfg['sequence_length']['use_length']
    regs = cfg['regularization_defaults']

    # 1) Input + slicing
    inputs = keras.layers.Input(
        shape=(cfg['sequence_length']['input_length'], 8)
    )

    x = global_norm(inputs)
    x = slice_last_n_timesteps(seq_len)(x)

    # 2) LSTM stack with your tuned regularization
    for i in range(arch['num_layers']):
        units  = arch['recurrent_units'][i]
        dr     = regs['dropout'][i]
        rdr    = regs['recurrent_dropout'][i]
        bn     = regs['batch_norm'][i]
        ln     = regs.get('layer_norm', [False]*arch['num_layers'])[i]
        return_sequences = (i < arch['num_layers'] - 1)

        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dr,
            recurrent_dropout=rdr,
            name=f'lstm_{i}'
        )(x)

        if bn:
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        if ln:
            x = layers.LayerNormalization(name=f'layer_norm_{i}')(x)

    # 3) Output heads
    loc = layers.Dense(2, activation='linear', name='loc_output')(x)
    ang = layers.Dense(2, activation='tanh',   name='angle_output')(x)

    # 4) Tune optimization hyperparameters
    # 4a) Base learning rate
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="LOG")

    # 4b) Scheduler choice
    sched_choice = hp.Choice("scheduler", ["none", "cosine", "exponential"])
    if sched_choice == "cosine":
        decay_steps = hp.Int("decay_steps_cosine", 374*5, 374*50, step=374)
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps
        )
    elif sched_choice == "exponential":
        decay_steps = hp.Int("decay_steps_exponential", 374*5, 374*50, step=374)
        er = hp.Float("decay_rate_exp",    0.80,      0.99,        step=0.01)
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate = er
        )
    else:
        schedule = lr  # constant learning rate

    # 4c) Optimizer choice
    opt_choice = hp.Choice("optimizer", ["adam", "rmsprop", "sgd", "nadam"])
    if opt_choice == "adam":
        optimizer = keras.optimizers.Adam(schedule)
    elif opt_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(schedule)
    elif opt_choice == "sgd":
        optimizer = keras.optimizers.SGD(schedule, momentum=0.9)
    else:
        optimizer = keras.optimizers.Nadam(schedule)

    # 5) Compile
    model = Model(inputs=inputs, outputs=[loc, ang])
    model.compile(optimizer=optimizer,
                  loss={'loc_output':'mse', 'angle_output':'mse'})
    return model


def build_stage4_model(hp):
    """
    Stage 4: Sequence‐Length Tuning
    - Tunes only the window length (how many timesteps to feed into the RNN)
    - Uses the best architecture, regularization, and optimization settings
      already stored in GLOBAL_CONFIG.
    """
    cfg  = GLOBAL_CONFIG
    arch = cfg["best_architecture"]
    regs = cfg["regularization_defaults"]
    opt  = cfg["optimization_defaults"]

    # 1) Tune how many of the last timesteps to use
    seq_choices = [5, 10, 15, 20, 25]
    seq_len = hp.Choice("sequence_length", values=seq_choices)

    # 2) Raw input is still the full buffer size
    inputs = keras.layers.Input(
        shape=(cfg["sequence_length"]["input_length"], 8)
    )

    # 3) Slice down to only the last `seq_len` steps

    x = global_norm(inputs)
    x = slice_last_n_timesteps(seq_len)(x)



    # 5) Rebuild LSTM stack with your saved regularization
    for i in range(arch["num_layers"]):
        units           = arch["recurrent_units"][i]
        dropout_rate    = regs["dropout"][i]
        rec_dropout_rate= regs["recurrent_dropout"][i]
        use_bn          = regs["batch_norm"][i]
        use_ln          = regs.get("layer_norm", [False]*arch["num_layers"])[i]

        return_sequences = (i < arch["num_layers"] - 1)
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=rec_dropout_rate,
            name=f"lstm_{i}"
        )(x)

        if use_bn:
            x = layers.BatchNormalization(name=f"batch_norm_{i}")(x)
        if use_ln:
            x = layers.LayerNormalization(name=f"layer_norm_{i}")(x)

    # 6) Outputs
    loc_output   = layers.Dense(2, activation="linear", name="loc_output")(x)
    angle_output = layers.Dense(2, activation="tanh",   name="angle_output")(x)

    # 7) Reconstruct your optimizer + schedule from the saved config
    lr    = opt["lr"]
    sched = None
    if opt.get("scheduler") == "cosine":
        sched = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=opt["decay_steps"]
        )
    elif opt.get("scheduler") == "exponential":
        sched = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=opt["decay_steps"],
            decay_rate=opt.get("decay_rate", 0.96)
        )
    else:
        sched = lr  # constant LR

    optim_name = opt.get("optimizer", "adam")
    if optim_name == "adam":
        optimizer = keras.optimizers.Adam(sched)
    elif optim_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(sched)
    elif optim_name == "sgd":
        optimizer = keras.optimizers.SGD(sched, momentum=0.9)
    else:
        optimizer = keras.optimizers.Nadam(sched)

    # 8) Compile and return
    model = Model(inputs=inputs, outputs=[loc_output, angle_output])
    model.compile(
        optimizer=optimizer,
        loss={"loc_output": "mse", "angle_output": "mse"}
    )
    return model


# Final Model: Consolidation
def build_final_model():
    """
    Final Model: Consolidation
    Pulls in every tuned setting:
      - best_architecture
      - sequence_length.use_length as input shape
      - regularization_defaults
      - optimization_defaults
    """
    cfg  = GLOBAL_CONFIG
    arch = cfg['best_architecture']
    seq  = cfg['sequence_length']['use_length']
    regs = cfg['regularization_defaults']
    opt  = cfg['optimization_defaults']

    # 1) Input shape uses only the tuned window length
    inputs = keras.layers.Input(shape=(seq, 8), name='input')

    # # 2) Optional: feature normalization
    # norm = clone_norm()
    x = global_norm(inputs)

    # x = inputs



    # 3) LSTM stack with saved regularization
    for i in range(arch['num_layers']):
        units    = arch['recurrent_units'][i]
        dr       = regs['dropout'][i]
        rdr      = regs['recurrent_dropout'][i]
        bn       = regs.get('batch_norm', [False]*arch['num_layers'])[i]
        ln       = regs.get('layer_norm', [False]*arch['num_layers'])[i]
        return_seq = (i < arch['num_layers'] - 1)

        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=dr,
            recurrent_dropout=rdr,
            name=f'lstm_{i}'
        )(x)
        if bn:
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        if ln:
            x = layers.LayerNormalization(name=f'layer_norm_{i}')(x)

    # 4) Output heads
    loc_out   = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_out = layers.Dense(2, activation='tanh',   name='angle_output')(x)

    # 5) Rebuild optimizer + schedule from config
    lr   = opt['lr']
    sched_type = opt.get('scheduler', 'none')
    if sched_type == 'cosine':
        decay_steps = opt['decay_steps']
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps
        )
    elif sched_type == 'exponential':
        decay_steps = opt['decay_steps']
        decay_rate  = opt.get('decay_rate', 0.96)
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    else:
        schedule = lr

    opt_name = opt.get('optimizer', 'adam')
    if opt_name == 'adam':
        optimizer = keras.optimizers.Adam(schedule)
    elif opt_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(schedule)
    elif opt_name == 'sgd':
        optimizer = keras.optimizers.SGD(schedule, momentum=0.9)
    else:
        optimizer = keras.optimizers.Nadam(schedule)

    # 6) Compile
    model = Model(inputs=inputs, outputs=[loc_out, angle_out])
    model.compile(
        optimizer=optimizer,
        loss={'loc_output':'mse','angle_output':'mse'}
    )
    return model
