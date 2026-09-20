


import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from keras import layers



# Define a cosine similarity–based loss function for headings
def cosine_loss(y_true, y_pred):
    # Use tf.keras.losses.CosineSimilarity which returns negative values (max similarity = -1)
    cos_sim = tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred)
    # Convert to a loss (0 when identical, higher when misaligned)
    return 1 + cos_sim  # When vectors are identical, cos_sim = -1, so loss becomes 0







# Define a hypermodel function
def build_model(hp):
    inputs = keras.Input(shape=(8,))

    # Hyperparameter: number of layers (between 1 and 3)
    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)
    x = inputs
    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        activation = hp.Choice(f"activation_{i}", values=["relu", "tanh", "elu","leaky_relu"])
        init = hp.Choice(f"kernel_init_{i}", values=["glorot_uniform", "he_normal"])
        reg = hp.Float(f"kernel_reg_{i}", min_value=1e-5, max_value=1e-2, sampling="LOG", default=1e-4)

        if activation == "leaky_relu":
            x = layers.Dense(units, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
            x = layers.LeakyReLU(alpha=0.1)(x)
        else:
            x = layers.Dense(units, activation=activation, kernel_initializer=init,
                             kernel_regularizer=tf.keras.regularizers.l2(reg))(x)

        # Optionally, add batch normalization
        if hp.Boolean(f"batch_norm_{i}"):
            x = layers.BatchNormalization()(x)
        # Optionally, add dropout
        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    loc_output = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = layers.Dense(2, activation='tanh', name='angle_output')(x)

    model = Model(inputs=[inputs], outputs=[loc_output, angle_output])

    # Hyperparameter: learning rate for the optimizer
    lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="LOG")
    # Hyperparameter: optimizer choice
    optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop", "nadam"])
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = keras.optimizers.Nadam(learning_rate=lr)


    # Hyperparameter for selecting the heading loss function:
    heading_loss_type = hp.Choice("heading_loss", values=["mse", "cosine"])
    if heading_loss_type == "mse":
        heading_loss = "mse"
    else:
        heading_loss = cosine_loss

    model.compile(optimizer= optimizer, loss=['mean_squared_error', heading_loss], loss_weights= [1.0, 1.0])
    return model



def build_default_mlp():
    from kerastuner import HyperParameters

    # Create a HyperParameters object and set fixed values.
    hp = HyperParameters()
    hp.Fixed("num_layers", 1)
    hp.Fixed("units_0", 100)                  # Single hidden layer of 100 neurons
    hp.Fixed("activation_0", "relu")          # Default activation is ReLU
    hp.Fixed("kernel_init_0", "glorot_uniform")
    hp.Fixed("kernel_reg_0", 1e-4)             # Equivalent to alpha in scikit-learn
    hp.Fixed("batch_norm_0", False)
    hp.Fixed("dropout_0", 0.0)
    hp.Fixed("lr", 1e-3)                      # Learning rate of 0.001
    hp.Fixed("optimizer", "adam")             # Adam is the default solver in scikit-learn
    hp.Fixed("heading_loss", "mse")           # Use MSE for the heading output loss

    # Now build the model with these hyperparameters
    model = build_model(hp)

    return model
