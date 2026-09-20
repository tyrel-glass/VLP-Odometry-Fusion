from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import os
import pickle
from keras.models import load_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import paper2_navigation.data.sensor_processing as c

from paper2_navigation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, VLP_MODELS_DIR

from keras.losses import MeanSquaredError

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization



vlp_dataset_path  = RAW_DATA_DIR / "cnc"
output_path  = VLP_MODELS_DIR / 'CNC/'

models_filename = output_path / "CNC_vlp_models.pkl"

app = typer.Typer()

@app.command()
def main(

):
    # -----------------------------------------
    logger.info("Reading vlp dataset")

    train_df = pd.read_csv(vlp_dataset_path/'cnc_fingerprint_01.csv', delimiter = '|')
    train_df = c.process_cnc(train_df)
    train_df = c.process_vlp(train_df)

    valid_df = pd.read_csv(vlp_dataset_path/'cnc_fingerprint_02.csv', delimiter = '|')
    valid_df = c.process_cnc(valid_df)
    valid_df = c.process_vlp(valid_df)


    X_train = train_df[['L1', 'L2', 'L3', 'L4']]
    y_train = train_df[['cnc_x', 'cnc_y']]

    X_valid = valid_df[['L1', 'L2', 'L3', 'L4']]
    y_valid = valid_df[['cnc_x', 'cnc_y']]

    model_high_acc_path = "high_acc_model.keras"
    mlp_high_acc_model, mlp_high_acc_scaler  = build_and_train_model(X_train, y_train, X_valid, y_valid,partial_data=False)
    mlp_high_acc_model.save(output_path / model_high_acc_path, save_format = 'tf' )

    # model_low_acc_path = "low_acc_model.keras"
    # mlp_low_acc_model, mlp_low_acc_scaler = build_and_train_model(X_train, y_train, X_valid, y_valid,partial_data=True)
    # mlp_low_acc_model.save(output_path / model_low_acc_path, save_format = 'tf')

    model_dic = {
        # 'low_acc': {'scaler': mlp_low_acc_scaler, 'model_path': model_low_acc_path},
        'high_acc': {'scaler': mlp_high_acc_scaler, 'model_path': model_high_acc_path}
    }

    pickle.dump(model_dic, open(models_filename, "wb"))

    logger.success("Created VLP models")
    # -----------------------------------------


def build_and_train_model(X_train, y_train, X_valid, y_valid,partial_data = False):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Define an exponential decay schedule
    def lr_schedule(epoch, initial_lr=0.01, decay_rate=0.995):
        return initial_lr * (decay_rate ** epoch)
    # Define the Learning Rate Scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Build a simple Keras model
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train_scaled.shape[1],kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(2, activation='linear')  # Output layer for 2 regression outputs
    ])
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[MeanSquaredError()])

    # Configure early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=30, restore_best_weights=True, verbose=1
    )

    if partial_data == False:
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_valid_scaled, y_valid),
            epochs=10000,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
    else:
        X_train_partial, _, y_train_partial, _ = train_test_split(
        X_train_scaled, y_train, train_size=0.2, random_state=42
            )
        history = model.fit(
            X_train_partial, y_train_partial,
            validation_data=(X_valid_scaled, y_valid),
            epochs=10000,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

    return  model, scaler

# Create a pipeline-like return object
class CustomPipeline:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def load_vlp_models(models_filename = models_filename, output_path = output_path):

    # Ensure paths are Path objects
    models_filename = Path(models_filename)
    output_path = Path(output_path)

    # Load the model dictionary
    try:
        model_dic = pickle.load(open(models_filename, "rb"))
    except Exception as e:
        raise ValueError(f"Error loading model dictionary from {models_filename}: {e}")

    # Reconstruct the pipelines
    pipelines = {}
    for key, components in model_dic.items():
        try:
            scaler = components['scaler']
            model_path = output_path / components['model_path']
            model = load_model(model_path)
            pipelines[key] = CustomPipeline(scaler, model)
        except Exception as e:
            raise ValueError(f"Error reconstructing pipeline for '{key}': {e}")

    return pipelines


if __name__ == "__main__":
    app()
