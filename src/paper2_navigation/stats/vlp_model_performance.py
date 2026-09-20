from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np

from paper2_navigation.config import EXTERNAL_DATA_DIR, TABLES_DIR

import paper2_navigation.data.gen_simulation_vlp_model as gvm
import paper2_navigation.data.odometer_path_navigation as pg
vlp_dataset_path = EXTERNAL_DATA_DIR / "vlp_dataset.csv"

app = typer.Typer()


@app.command()
def main(

):
    df = pd.read_csv(vlp_dataset_path, index_col=0)
    vlp_model_dic = pg.read_vlp_models()
    X = df.iloc[:, :11]
    y = df.iloc[:, 11:]

    name_lst = []
    err_mean_lst = []
    err_std_lst = []
    num_training_points_lst = []
    per_90_lst = []


    for vlp_name, vlp_model in vlp_model_dic.items():
        y_pre = vlp_model.predict(X)
        errs = np.sqrt(
            np.square(y_pre[:, 0] - y.values[:, 0]) + np.square(y_pre[:, 1] - y.values[:, 1])
        )

        num_train_pts = gvm.model_training_samples_dic[vlp_name]
        num_training_points_lst.append(num_train_pts)

        name_lst.append(vlp_name.split('_')[0])
        err_mean_lst.append(np.mean(errs))
        err_std_lst.append(np.std(errs))
        per_90_lst.append(np.percentile(errs, 90))


    res_df = pd.DataFrame({
        'VLP model accuracy':name_lst,
        'Training points':num_training_points_lst,
        'Mean error (m)':err_mean_lst,
        'Standard Deviation (m)': err_std_lst,
        '90th percentile (m)': per_90_lst

    })

    res_df.set_index('VLP model accuracy', inplace=True)

    res_df = res_df.T
    res_df.to_latex(TABLES_DIR/'vlp_performance.tex')






if __name__ == "__main__":
    app()
