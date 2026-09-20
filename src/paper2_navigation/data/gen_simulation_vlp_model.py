from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import os
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from paper2_navigation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()

model_training_samples_dic = {
    'low_acc':40,
    'med_acc':400,
    'high_acc':4000
}


@app.command()
def main(
    vlp_dataset_path: Path = EXTERNAL_DATA_DIR / "vlp_dataset.csv",
    output_path: Path = VLP_MODELS_DIR,
):
    # -----------------------------------------
    logger.info("Reading vlp dataset")
    df = pd.read_csv(vlp_dataset_path, index_col=0)

    model_dic = {}
    for model_name, num_samples in model_training_samples_dic.items():
        model_dic[model_name] = build_model(df= df, train_samples=num_samples)
    # model_dic = {
    #     "low_acc":build_model(df=df, train_samples=40 ),
    #     "med_acc": build_model(df=df, train_samples=400),
    #     "high_acc":build_model(df=df, train_samples=4000)
    # }


    models_filename = output_path / "vlp_models.pkl"
    pickle.dump(model_dic, open(models_filename, "wb"))
    logger.success("Created VLP models")
    # -----------------------------------------


def build_model(df, train_samples: int):
    X = df.iloc[:, :11]
    y = df.iloc[:, 11:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, random_state=42
    )

    regr = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(max_iter=1500))])
    regr.fit(X_train.values, y_train.values)

    y_pre = regr.predict(X_test.values)
    errs = np.sqrt(
        np.square(y_pre[:, 0] - y_test.values[:, 0]) + np.square(y_pre[:, 1] - y_test.values[:, 1])
    )
    logger.info("mean error of: " + str(errs.mean()) + str(train_samples)+" training samples")

    return regr



if __name__ == "__main__":
    app()
