from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import pickle
import numpy as np
import keras
import paper2_navigation.data.preprocessing as p
import paper2_navigation.data.path_geometry as path_geometry
import matplotlib.pyplot as plt

from paper2_navigation.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, FIGURES_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Performing inference for model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Inference complete.")
    # -----------------------------------------


    # plot_acc_grid()
    # plot_err_grid()
    # plot_nav_points()
    # plot_encoder_err()
    plot_vlp_err()

# ============================ PLOT TO SHOW ENCODER DRIFT====================================
def plot_encoder_err():
    plot_params = {
        'start_index':0,
        'path_length':113,
        'filter_params':{
            'n':'n_4',
            'direction':'_clockwise',
            'vlp_acc':'low_acc',
            'odo_acc':'err_1',
            'run':'run1'
        }
    }
    fig, axs  = plt.subplots(ncols = 3, nrows = 1, figsize = (10,3), layout = 'constrained')
    odo_acc = ['err_1', 'err_2', 'err_3']
    start_index = plot_params['start_index']
    path_length = plot_params['path_length']
    end_index = start_index + path_length
    X_data, y_data = load_run_data(plot_params['filter_params'])
    x_tar = X_data[start_index:end_index,0]
    y_tar = X_data[start_index:end_index,1]
    heading_tar = X_data[start_index:end_index,2]
    # axs[2].plot(heading_tar, label = 'Desired heading')
    axs[0].plot(x_tar, y_tar, label = 'Desired path')
    for i in range(3):
        plot_params['filter_params']['odo_acc'] = odo_acc[i]
        X_data, y_data = load_run_data(plot_params['filter_params'])
        axs[0].plot(y_data[start_index:end_index,0], y_data[start_index:end_index,1], label = odo_acc[i])
        ang_diff = heading_tar - y_data[start_index:end_index,2]

        ang_diff = (ang_diff + 180) % 360 - 180
        axs[2].plot(ang_diff, label = odo_acc[i])

        loc_err = np.sqrt(np.square(x_tar - y_data[start_index:end_index,0]) + np.square(y_tar - y_data[start_index:end_index,1]))
        axs[1].plot(loc_err, label = odo_acc[i])
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()



    fig.savefig(FIGURES_DIR/'encoder_drift')

# ======================================================================================

# ============================ PLOT TO SHOW VLP DRIFT====================================
def plot_vlp_err():
    plot_params = {
        'start_index':0,
        'path_length':113,
        'filter_params':{
            'n':'n_4',
            'direction':'_clockwise',
            'vlp_acc':'low_acc',
            'odo_acc':'err_1',
            'run':'run1'
        }
    }
    fig, axs  = plt.subplots(ncols = 2, nrows = 1, figsize = (10,3), layout = 'constrained')
    vlp_acc = ['low_acc', 'med_acc', 'high_acc']
    start_index = plot_params['start_index']
    path_length = plot_params['path_length']
    end_index = start_index + path_length
    X_data, y_data = load_run_data(plot_params['filter_params'])
    x_tar = y_data[start_index:end_index,0]
    y_tar = y_data[start_index:end_index,1]
    axs[0].plot(x_tar, y_tar, label = 'Desired path')
    for i in range(3):
        plot_params['filter_params']['vlp_acc'] = vlp_acc[i]
        X_data, y_data = load_run_data(plot_params['filter_params'])
        axs[0].plot(X_data[start_index:end_index,3], X_data[start_index:end_index,4], label = vlp_acc[i])


        loc_err = np.sqrt(np.square(x_tar - X_data[start_index:end_index,3]) + np.square(y_tar - X_data[start_index:end_index,4]))
        axs[1].plot(loc_err, label = vlp_acc[i])
    axs[0].legend()
    axs[1].legend()




    fig.savefig(FIGURES_DIR/'vlp_drift')

# ======================================================================================


def load_run_data(filter_params):
    test_file = p.path_filter(filter_params, mode = 'include')[0]
    X_data, y_data = p.load_data([test_file])
    return X_data[0], y_data[0]


def plot_path(plot_params, ax):
    start_index = plot_params['start_index']
    path_length = plot_params['path_length']
    end_index = start_index + path_length
    X_data, y_data = load_run_data(plot_params['filter_params'])

    ax.plot(X_data[start_index:end_index,0],X_data[start_index:end_index,1], label = 'desired path')
    ax.plot(X_data[start_index:end_index,3],X_data[start_index:end_index,4], label = 'vlp estimate')
    ax.plot(y_data[start_index:end_index,0],y_data[start_index:end_index,1], label = 'path taken')

    ax.legend()

def calc_encoder_err(X,y):
    err = np.square(X[:,0] - y[:,0]) + np.square(X[:,1] - y[:,1])
    return err
def calc_vlp_err(X,y):
    err = np.square(X[:,3] - y[:,0]) + np.square(X[:,4] - y[:,1])
    return err




def plot_nav_points():
    fig, axs  = plt.subplots(ncols = 3, nrows = 1, figsize = (5,2), layout = 'constrained', sharey=True)
    n_lst = [2, 4, 8]
    for row in range(3):
        ax = axs[row]
        n = n_lst[row]

        ax.set_ylim(0,6)
        ax.set_xlim(0,6)
        points = path_geometry.create_polygon_targets(n)
        ax.scatter(points[:,0], points[:,1])
    fig.savefig(FIGURES_DIR/'nav_points_ex')


def plot_acc_grid():
    plot_params = {
        'start_index':0,
        'path_length':100,
        'filter_params':{
            'n':'n_4',
            'direction':'_clockwise',
            'vlp_acc':'low_acc',
            'odo_acc':'err_1',
            'run':'run1'
        }
    }
    fig, axs  = plt.subplots(ncols = 3, nrows = 3, figsize = (15,15), layout = 'constrained', sharey = True, sharex = True)
    vlp_acc = ['err_1', 'err_2', 'err_3']
    odo_acc = ['low_acc', 'med_acc', 'high_acc']
    for row in range(3):
        for col in range(3):
            plot_params['filter_params']['vlp_acc'] = vlp_acc[row]
            plot_params['filter_params']['odo_acc'] = odo_acc[col]
            ax = axs[row, col]
            plot_path(plot_params, ax)
    fig.savefig(FIGURES_DIR/'acc_grid')





def plot_position_error(plot_params, ax):
    start_index = plot_params['start_index']
    path_length = plot_params['path_length']
    end_index = start_index + path_length
    X_data, y_data = load_run_data(plot_params['filter_params'])

    encoder_err = calc_encoder_err(X_data, y_data)
    vlp_err = calc_vlp_err(X_data, y_data)

    ax.plot(encoder_err, label = 'encoder error')
    ax.plot(vlp_err, label = 'VLP error')
    ax.legend()

def plot_err_grid():
    plot_params = {
        'start_index':38,
        'path_length':113,
        'filter_params':{
            'n':'n_4',
            'direction':'_clockwise',
            'vlp_acc':'low_acc',
            'odo_acc':'err_1',
            'run':'run1'
        }
    }
    fig, axs  = plt.subplots(ncols = 3, nrows = 3, figsize = (15,15), layout = 'constrained', sharey= True)
    vlp_acc = ['err_1', 'err_2', 'err_3']
    odo_acc = ['low_acc', 'med_acc', 'high_acc']
    for row in range(3):
        for col in range(3):
            plot_params['filter_params']['vlp_acc'] = vlp_acc[row]
            plot_params['filter_params']['odo_acc'] = odo_acc[col]
            ax = axs[row, col]
            plot_position_error(plot_params, ax)
    print('SAVING FIGUREE!!!!')
    fig.savefig(FIGURES_DIR/'err_grid')
    print('FIGUREE SAVED!!!!')





    # X_labels = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist", "vlp_x_hist", "vlp_y_hist"]
    # y_labels = ["x_hist", "y_hist", "heading_hist"]


if __name__ == "__main__":
    app()
