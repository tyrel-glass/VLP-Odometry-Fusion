from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from paper2_navigation.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
import pickle
import numpy as np
import keras
import paper2_navigation.data.preprocessing as p
import matplotlib.pyplot as plt



app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # -----------------------------------------
):
    plot_params = {
        'model_name':'model_03.keras',
        'name':'pre_model_ex1',
        'start_index':0,
        'path_length':50,
        'filter_params':{
            'n':'n_2',
            'direction':'_clockwise',
            'vlp_acc':'med_acc',
            'odo_acc':'err_2',
            'run':'run1'
        }
    }
    plot_model_path_predictions(plot_params=plot_params)

    plot_params = {
        'model_name':'model_03.keras',
        'name':'pre_model_ex2',
        'start_index':0,
        'path_length':113,
        'filter_params':{
            'n':'n_4',
            'direction':'_clockwise',
            'vlp_acc':'med_acc',
            'odo_acc':'err_2',
            'run':'run1'
        }
    }
    plot_model_path_predictions(plot_params=plot_params)

    plot_params = {
        'model_name':'model_03.keras',
        'name':'pre_model_ex3',
        'start_index':0,
        'path_length':213,
        'filter_params':{
            'n':'n_8',
            'direction':'shuffle',
            'vlp_acc':'med_acc',
            'odo_acc':'err_2',
            'run':'run1'
        }
    }
    plot_model_path_predictions(plot_params=plot_params)



def load_run_data(filter_params):
    """
    Takes a filter_param dict and uses it to load all paths matching the requirements
    """
    test_files = p.path_filter(filter_params, mode = 'include')
    X_data, y_data = p.load_data(test_files)
    return X_data, y_data

def window_every_sample(arr, win_len):
    """
    Takes an array and generates a sliding window of win_len and stride of 1
    """
    lst = []
    for i in range(len(arr) - win_len + 1):
        lst.append(arr[i:i+win_len, :])
    X_all = np.array(lst)
    return X_all

def load_model(model_name = 'model_02.keras'):
    model = keras.models.load_model(MODELS_DIR / model_name)
    return model

def min_ang_diff(ang1, ang2):
    ang_diff = ang1 - ang2
    min_ang_diff = ((ang_diff + 180)%360) - 180
    return min_ang_diff

def calc_dist_err(x1, y1, x2, y2):
    return np.sqrt(np.square(x1-x2) + np.square(y1-y2))



def plot_model_path_predictions(plot_params):
    win_len = 10
    X_data_all_runs, y_data_all_runs = load_run_data(plot_params['filter_params'])  # pull target runs
    X_data = X_data_all_runs[0] # take first matching run
    y_data = y_data_all_runs[0]
    X_all = window_every_sample(X_data, win_len = win_len) # create windows with stride 1
    X_all_scaled = p.apply_scaler(X_all, p.build_scaler())
    model = load_model(model_name= plot_params['model_name'])
    pre = model.predict(X_all_scaled)

    x_model_est = pre[0][:,0]
    y_model_est = pre[0][:,1]
    model_ang_est = p.vector_to_ang(pre[1], unit = 'degrees').numpy()

    real_ang = y_data[9:,2]
    odo_ang_est = X_all[:,-1,2]

    model_ang_err = min_ang_diff(real_ang, model_ang_est)
    odo_ang_err = min_ang_diff(real_ang, odo_ang_est )

    b = plot_params['start_index']
    e = plot_params['path_length'] + b

    x_r = y_data[9:,0]
    y_r = y_data[9:,1]
    x_o = X_data[9:,0]
    y_o = X_data[9:,1]
    x_m = x_model_est
    y_m = y_model_est

    odo_pos_err = calc_dist_err(x_r,y_r,x_o,y_o)
    model_pos_err = calc_dist_err(x_r,y_r,x_m,y_m)

    fig, axs  = plt.subplots(ncols = 3, nrows = 1, figsize = (10,3), layout = 'constrained')

    axs[0].plot(X_data[b:e,0], X_data[b:e,1], label = 'odometer est')
    axs[0].plot(y_data[b:e,0], y_data[b:e,1], marker = '.', label = 'real')
    axs[0].plot(x_model_est[b:e - 10], y_model_est[b:e - 10], marker = '.', label = 'model')
    axs[0].legend()
    axs[0].set_title('Predicted vs real path')

    axs[1].plot(np.abs(model_ang_err), label = 'Model heading error')
    axs[1].plot(np.abs(odo_ang_err),label = 'Odometer heading error' )
    axs[1].legend()
    axs[1].set_title('Error in heading estiamte')

    axs[2].plot(model_pos_err, label = 'Model error')
    axs[2].plot(odo_pos_err, label = 'encoder error')
    axs[2].legend()
    axs[2].set_title('Error in position estimate')

    fig.savefig(FIGURES_DIR/plot_params['name'])


if __name__ == "__main__":
    app()
