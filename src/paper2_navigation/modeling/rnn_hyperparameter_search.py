from pathlib import Path
from paper2_navigation.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TRAINING_LOGS_DIR, INTERIM_DATA_DIR
import typer
from loguru import logger
from tqdm import tqdm

import tensorflow as tf
import keras
import keras_tuner as kt

from paper2_navigation.config import MODELS_DIR, PROCESSED_DATA_DIR

import paper2_navigation.modeling.rnn as rnn

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


def build_random_search_tuner(directory, project_name, overwrite):
    random_search_tuner = kt.RandomSearch(
        rnn.build_model,
        objective='val_loss',
        max_trials = 100,
        overwrite = overwrite,
        directory = directory,
        project_name = project_name,
        seed = 42
    )
    return random_search_tuner

def run_hyperparameter_tuner():

    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    random_search_tuner = build_random_search_tuner(
        TRAINING_LOGS_DIR,
        'rnn_rnd_search',
        overwrite= False
        )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(TRAINING_LOGS_DIR / 'tensorboard')
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)

    random_search_tuner.search(x = data['X_train'],
                               y = [data['y_train'][:,[0,1]],  p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
                               epochs = 10,
                               validation_data = (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()]),
                               callbacks = [early_stopping_cb, tensorboard_cb],
                               batch_size = 512
                               )


    best_model = random_search_tuner.get_best_models(num_models=1)[0]
    best_model.save(MODELS_DIR / 'model_02.keras')

if __name__ == "__main__":
    app()