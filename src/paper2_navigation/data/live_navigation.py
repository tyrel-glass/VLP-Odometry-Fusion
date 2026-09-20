from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import keras
import numpy as np
import pandas as pd
import pickle
import paper2_navigation.robot as r

from paper2_navigation.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR

import paper2_navigation.data.preprocessing as p
import paper2_navigation.data.odometer_path_navigation as opn



ERROR_VALUES = {'err_1': 0.01, 'err_2': 0.05, 'err_3': 0.1}
PATH_REPEATS = 1





MODEL = 'model_02.keras'
def ang_loss_fn(y_true, y_pred):
    return keras.losses.cosine_similarity(y_true, y_pred) + 1


X_scaler = p.build_scaler()


app = typer.Typer()


@app.command()
def main(
    model_name,
    navigation_paths,
    output_dataset_folder_name,

):
    #-------- Load navigation paths ---------
    GENERATED_PATHS_DIR = INTERIM_DATA_DIR/navigation_paths
    with open(GENERATED_PATHS_DIR, 'rb') as f:
        generated_paths = pickle.load(f)

    #-------- Load rnn model --------
    model = keras.models.load_model(MODELS_DIR / model_name,custom_objects={"ang_loss_fn": ang_loss_fn})

    #--------- Read in VLP dataset ------
    logger.info("Reading in VLP dataset")
    vlp_dataset_path = EXTERNAL_DATA_DIR / "vlp_dataset.csv"
    df = pd.read_csv(vlp_dataset_path, index_col=0)
    logger.success('Pulled in VLP dataset')

    #--------- Set output folder--------
    output_datset_folder_path = INTERIM_DATA_DIR / output_dataset_folder_name
    output_datset_folder_path.mkdir(parents = False, exist_ok = True)


    #---------- Read in VLP models -------
    vlp_models = opn.read_vlp_models()




    filter_params = {
        # 'n':'n_2',
        # 'direction':'shuffle',
        # 'radius':'rad_1-5'
    }

    target_paths = filter_path_names(filter_params, generated_paths)

    # for vlp_name, vlp_model in vlp_models.items():

    #     for err_name, err_val in ERROR_VALUES.items():

    # for run_no in range(PATH_REPEATS):

    for path_name, path_coordinates in target_paths.items():
        logger.info(f'Starting run {path_name}')
        start_x = 1
        start_y = 1
        start_heading = 45
        vlp_name = 'med_acc'
        vlp_model = vlp_models[vlp_name]
        err_name = 'err_2'
        err_val = ERROR_VALUES[err_name]

        run_no = 0

        robot = r.Robot(x=start_x, y=start_y, heading =start_heading, step_err = err_val, turn_err = err_val, df = df, vlp_mod = vlp_model,navigation_method = 'odometer')
        navigate_to_target(robot, x = 5, y = 5, model = model)
        num_repeats = int(20 / len(path_coordinates) ) # try and get all pahts to navigate 20 points
        for _ in range(num_repeats):
            for i in range(len(path_coordinates)):
                x = path_coordinates[i,0]
                y = path_coordinates[i,1]
                navigate_to_target(robot, x = x, y = y, model = model)

        name = path_name +f'_{vlp_name}_{err_name}_run{run_no}'

        X_data = np.array([robot.encoder_x_hist, robot.encoder_y_hist, robot.encoder_heading_hist, robot.vlp_x_hist, robot.vlp_y_hist, robot.vlp_heading_hist]).T
        y_data = np.array([robot.x_hist, robot.y_hist, robot.heading_hist]).T
        model_data = np.array([robot.model_x_hist, robot.model_y_hist, robot.model_heading_hist]).T

        data_dic = {'X':X_data, 'y':y_data, 'm':model_data}

        file_path = output_datset_folder_path / name
        pickle.dump(data_dic, open(file_path , 'wb'))


def make_prediction(X, model):
    return model.predict(np.expand_dims(X,0), verbose = None)

def filter_path_names(filter_parameters, paths_dict):
    """Takes a dict of filter parameters and uses them to select matching names from a dictionary of paths"""
    parameter_list = list(filter_parameters.values())
    target_list = paths_dict.keys()  # start with all
    for parameter in parameter_list:
        target_list = [path_name for path_name in target_list if (parameter in path_name)]

    filtered_dict = {path_name : paths_dict[path_name] for path_name in target_list}
    return filtered_dict

def navigate_to_target(robot,x,y,model, step_size = 0.1, target_threshold = 0.1):
    robot.target_x = x
    robot.target_y = y
    window_len = model.input_shape[1]
    if window_len == None:
         window_len = 40

    robot.navigation_method = 'odometer'
    while (robot.calc_distance_to_target() > target_threshold) & opn.in_bounds(robot,x_lim=(0.5,6.5), y_lim = (0.5,5.7)):

        robot.correct_heading()
        robot.step(step_size)


        if len(robot.x_hist) >= window_len:
                # continue
                robot.navigation_method = 'model'

                X_data = np.array([robot.encoder_x_hist[-window_len:], robot.encoder_y_hist[-window_len:], robot.encoder_heading_hist[-window_len:], robot.vlp_x_hist[-window_len:], robot.vlp_y_hist[-window_len:], robot.vlp_heading_hist[-window_len:]]).T
                X_data = X_scaler.transform(X_data)

                pre = make_prediction(X = X_data, model = model)
                robot.model_x = pre[0][0,0]
                robot.model_y = pre[0][0,1]
                robot.model_heading = p.vector_to_ang(pre[1], unit = 'degrees').numpy()[0]
                robot.model_x_hist.append(robot.model_x)
                robot.model_y_hist.append(robot.model_y)
                robot.model_heading_hist.append(robot.model_heading)
        else:
            robot.model_heading_hist.append(robot.encoder_heading)
            robot.model_x_hist.append(robot.encoder_x)
            robot.model_y_hist.append(robot.encoder_y)


if __name__ == "__main__":
    app()
