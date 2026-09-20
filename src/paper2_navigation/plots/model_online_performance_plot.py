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
from paper2_navigation.modeling.rnn import ang_loss_fn

import paper2_navigation.plots.model_performance_plotting as pp


app = typer.Typer()

@app.command()
def main(
):
    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)
    model = pp.load_model(model_name = 'model_03.keras')
    x_win = data['X_valid']
    y_win = data['y_valid']

    test_model(x_win, y_win, model, FIGURES_DIR/'offline_model_test')




def test_model(x_win, y_win, model, plot_name):
    all_pre = model.predict(x_win)

    fig, axs  = plt.subplots(ncols = 2, nrows = 1, figsize = (10,3), layout = 'constrained')

    dist_errs = np.sqrt(np.square(all_pre[0][:,0] - y_win[:,0]) + np.square(all_pre[0][:,1] - y_win[:,1]))

    axs[0].hist(dist_errs, bins = 60, alpha = 0.3, label = 'model')

    odo_errs = np.sqrt(np.square(x_win[:,-1,0]*7 - y_win[:,0]) + np.square(x_win[:,-1,1]*7 - y_win[:,1]))


    axs[0].hist(odo_errs, bins = 60, alpha = 0.3, label = 'odometer')

    vlp_errs = np.sqrt(np.square(x_win[:,-1,3]*7 - y_win[:,0]) + np.square(x_win[:,-1,4]*7 - y_win[:,1]))

    axs[0].hist(vlp_errs, bins = 60, alpha = 0.3, label = 'vlp')

    axs[0].legend()
    axs[0].set_title('Distance errors')

    # ang_pre_mag = np.linalg.norm(all_pre[1], axis = 1)
    ang_real = y_win[:,2] # Actual robot heading

    ang_pre = p.vector_to_ang(all_pre[1], unit = 'degrees').numpy() # heading predicted by model

    model_angle_err = pp.min_ang_diff(ang_pre, ang_real)

    axs[1].hist(model_angle_err, bins = 70, alpha = 0.3, label = 'model angle error')

    ang_enc = x_win[:,-1,2]
    encoder_angle_err = pp.min_ang_diff(ang_enc, ang_real)

    axs[1].hist(encoder_angle_err, bins = 70, alpha = 0.3, label = 'encoder angle error')

    ang_vlp = x_win[:,-1,5]
    vlp_angle_err = pp.min_ang_diff(ang_vlp, ang_real)

    axs[1].hist(vlp_angle_err, bins = 70, alpha = 0.3, label = 'VLP angle error')


    axs[1].set_title('Heading errors')
    axs[1].legend()

    performance_stats = {}

    performance_stats['mean_vlp_angle_err'] = vlp_angle_err.mean()
    performance_stats['mean_encoder_angle_err'] = encoder_angle_err.mean()
    performance_stats['mean_model_angle_err'] = model_angle_err.mean()
    performance_stats['mean_vlp_distance_err'] = vlp_errs.mean()
    performance_stats['mean_odometer_distance_err'] = odo_errs.mean()
    performance_stats['mean_model_dist_err'] = dist_errs.mean()

    fig.savefig(plot_name)




if __name__ == "__main__":
    app()
