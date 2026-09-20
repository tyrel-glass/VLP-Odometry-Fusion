""""
This code creates a target paths, then generates robots with various error parameters to navigate these paths.
The resulting datasets are then stored in the INERM_DATA_DIR
"""

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from paper2_navigation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import pickle
import paper2_navigation.robot as r


    # Constants
ERROR_VALUES = {'err_1': 0.01, 'err_2': 0.05, 'err_3': 0.1}

PATH_REPEATS = 4

@app.command()
def main(
     input_path_file_name,
     output_folder_name

):

    # -------------------------------------------------------------
    GENERATED_PATHS_FILE_PATH = INTERIM_DATA_DIR/input_path_file_name
    OUTPUT_FOLDER_PATH = INTERIM_DATA_DIR / output_folder_name
    OUTPUT_FOLDER_PATH.mkdir(parents = False, exist_ok = True)


    logger.info("Reading in VLP dataset")
    vlp_dataset_path = EXTERNAL_DATA_DIR / "vlp_dataset.csv"
    df = pd.read_csv(vlp_dataset_path, index_col=0)
    logger.success('Pulled in VLP dataset')




    with open(GENERATED_PATHS_FILE_PATH, 'rb') as f:
        generated_paths = pickle.load(f)

    vlp_models = read_vlp_models()



    for vlp_name, vlp_model in vlp_models.items():

        for err_name, err_val in ERROR_VALUES.items():

            for run_no in range(PATH_REPEATS):

                for path_name, path_coordinates in generated_paths.items():

                            name = path_name +f'_{vlp_name}_{err_name}_run{run_no}'

                            robot = r.Robot(x=path_coordinates[0,0], y=path_coordinates[0,1], heading = 0, step_err = err_val, turn_err = err_val, df = df, vlp_mod = vlp_model)


                            for i in range(3):
                                for j in range(len(path_coordinates)):
                                    navigate_to_target(robot,path_coordinates[j,0],path_coordinates[j,1])

                            X_data = np.array([robot.encoder_x_hist, robot.encoder_y_hist, robot.encoder_heading_hist, robot.vlp_x_hist, robot.vlp_y_hist, robot.vlp_heading_hist]).T
                            y_data = np.array([robot.x_hist, robot.y_hist, robot.heading_hist]).T

                            data_dic = {'X':X_data, 'y':y_data}

                            file_path = OUTPUT_FOLDER_PATH / name
                            pickle.dump(data_dic, open(file_path , 'wb'))


    # -----------------------------------------


def read_vlp_models():
    """Reads in the vlp model"""

    vlp_models_path = VLP_MODELS_DIR / "vlp_models.pkl"
    return pickle.load(open(vlp_models_path, 'rb'))

def in_bounds(robot,x_lim, y_lim):
    return (robot.x > x_lim[0])&(robot.x < x_lim[1])&(robot.y > y_lim[0])&(robot.y < y_lim[1])

def navigate_to_target(robot,x,y,step_size = 0.1, target_threshold = 0.1):
    robot.target_x = x
    robot.target_y = y
    while (robot.calc_distance_to_target() > target_threshold) & in_bounds(robot,x_lim=(0.5,6.5), y_lim = (0.5,5.7)):
        robot.correct_heading()
        robot.step(step_size)



if __name__ == "__main__":
    app()
