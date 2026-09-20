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




@app.command()
def main(
    path_file_name
):

    GENERATED_PATHS_DIR = INTERIM_DATA_DIR/path_file_name
    DIRECTIONS = ['clockwise', 'anticlockwise', 'shuffle']
    N_VALUES = list(range(2, 11))
    TARGET_RADII = [1, 1.5, 2, 2.5]

    targets = generate_targets(n_values=N_VALUES,radii = TARGET_RADII, directions= DIRECTIONS, save_path =GENERATED_PATHS_DIR)



def generate_targets(n_values, radii, directions, save_path=None):
    """
    Generates and saves target points for given parameters.

    Parameters:
        n_values (list): List of values for the number of points per polygon.
        radii (list): List of radii to use for the circular target paths.
        directions (list): List of directions (e.g., 'clockwise', 'anticlockwise', 'shuffle').
        save_path (Path, optional): If provided, saves the targets to this file.

    Returns:
        dict: A dictionary containing target paths for each combination of parameters.
    """
    targets_dict = {}

    for n in n_values:
        for radius in radii:
            for direction in directions:
                key = f'n_{n}_rad_{str(radius).replace(".", "-")}_{direction}'
                targets = create_poly_targets(n, radius)

                if direction == 'anticlockwise':
                    targets = targets[::-1]
                elif direction == 'shuffle':
                    np.random.shuffle(targets)

                targets_dict[key] = targets

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(targets_dict, f)

    return targets_dict

def create_poly_targets(n, radius=2, center=(3.5, 3)):
    """Creates n points on the circumference of a circle with a specified radius and center."""
    return np.array([
        [radius * np.cos(ang) + center[0], radius * np.sin(ang) + center[1]]
        for ang in np.linspace(0, 2 * np.pi, n + 1)[:] + np.pi / 4
    ])


if __name__ == "__main__":
    app()
