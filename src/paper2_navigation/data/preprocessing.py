import random
import pickle
from keras import ops
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from paper2_navigation.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

import typer
from typing_extensions import Annotated
from typing import List
from loguru import logger
from tqdm import tqdm
from pathlib import Path

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def build_train_data(
        input_path_data_dir: str,
        dataset_save_name: str,
        exclude_model_data: bool = False

):

    include_model_data =  not exclude_model_data

    overlap = 0.975
    window_len = 10
    skip = [' ']


    data_dir = INTERIM_DATA_DIR / input_path_data_dir # pull from default spot


    ###########################################
    # CONFIG PARAMETERS (to be passed via cmd)
    ###########################################

    file_list = path_filter(skip,data_dir,  mode = 'exclude')

    train_files, valid_files, test_files = train_test_split_list(file_list)




    ##########################################

    X_scaler = build_scaler()

    X_train_data, y_train_data, m_train_data = load_data(train_files, include_model_data)
    X_valid_data, y_valid_data, m_valid_data = load_data(valid_files, include_model_data)
    X_test_data, y_test_data , m_test_data= load_data(test_files, include_model_data)

    X_train_window, y_train, m_train = create_windows(X_train_data, y_train_data,m_train_data, overlap, window_len)
    X_valid_window, y_valid, m_valid = create_windows(X_valid_data, y_valid_data, m_valid_data, overlap, window_len)
    X_test_window, y_test, m_test = create_windows(X_test_data, y_test_data, m_test_data, overlap, window_len)

    X_train = apply_scaler(X_train_window, X_scaler)
    X_valid = apply_scaler(X_valid_window, X_scaler)
    X_test = apply_scaler(X_test_window, X_scaler)

    data_dic = {
        'train_files':train_files,
        'valid_files':valid_files,
        'test_files':test_files,
        'X_train_data':X_train_data,
        'X_valid_data':X_valid_data,
        'X_test_data':X_test_data,
        'y_train_data':y_train_data,
        'y_valid_data':y_valid_data,
        'y_test_data':y_test_data,
        'm_train_data':m_train_data,
        'm_test_data':m_test_data,
        'm_valid_data':m_valid_data,
        'X_train':X_train,
        'X_valid':X_valid,
        'X_test':X_test,
        'y_train':y_train,
        'y_valid':y_valid,
        'y_test':y_test,
        'm_train':m_train,
        'm_valid':m_valid,
        'm_test':m_test
    }

    with open(PROCESSED_DATA_DIR/dataset_save_name, 'wb') as handle:
        pickle.dump(data_dic, handle)

def build_scaler():
    X_limits = np.array([
        [0,0,-180],
        [1,1,180]
    ])

    X_scaler = MinMaxScaler()
    X_scaler.fit(X_limits)

    return X_scaler

def apply_scaler(X_data, scaler = None):
    if scaler is None:
        scaler = build_scaler()
    num_windows, win_len, num_features = X_data.shape
    X_scaled = scaler.transform(X_data.reshape(-1,num_features))
    X_scaled = X_scaled.reshape(num_windows, win_len, num_features)
    return X_scaled







def train_test_split_list(lst, train_prop = 0.8, valid_prop = 0.2):
    """
    Takes a list, shuffles it, then breaks in into three parts
    """
    random.seed(42)
    random.shuffle(lst)
    train_prop = 0.6
    valid_prop = 0.2
    test_prop = 1 - (train_prop + valid_prop)
    num_train_elements = int(train_prop * len(lst))
    num_valid_elements = int(test_prop*len(lst))
    num_test_elements = int(valid_prop*len(lst))

    train_elements = [lst.pop() for i in range(num_train_elements)]
    valid_elements = [lst.pop() for i in range(num_valid_elements)]
    test_elements = [lst.pop() for i in range(len(lst))]

    return train_elements, valid_elements, test_elements


def load_data(file_list, include_model_data = False):
    x_lst = []
    y_lst = []
    m_lst = []

    for file_name in file_list:
        data = pickle.load(open(file_name, 'rb'))
        y_lst.append(data['y'])
        x_lst.append(data['X'])

        if include_model_data:
            m_lst.append(data['m'])
        else:
            m_lst.append(data['y'])  # just use the real values to fill the model predictions

    return x_lst, y_lst, m_lst


def create_windows(x, y, m , overlap = 0.5, window_len = 10):
    """

    """
    x_lst = []
    y_lst = []
    m_lst = []

    for i in range(len(x)):

        x_win, y_win, m_win = window_data(x[i], y[i], m[i], overlap = overlap ,window_len = window_len)

        if len(x_win) > window_len:
            x_lst.append(x_win)
            y_lst.append(y_win)
            m_lst.append(m_win)

    # X_labels = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist", "vlp_x_hist", "vlp_y_hist"]
    # y_labels = ["x_hist", "y_hist", "heading_hist"]

    X = np.concatenate(x_lst)
    y = np.concatenate(y_lst)

    m = np.concatenate(m_lst)

    return X , y, m

def window_data(X,y,m, overlap = 0.5 ,window_len = 10):
    """
    Used to split X,y arrays into windows
    """
    overlap_samples = int(overlap * window_len)
    start_index = 0
    end_index = window_len
    X_lst = []
    y_lst = []
    m_lst = []
    while end_index <= len(X):
        X_win = X[start_index:end_index]
        y_win = y[start_index:end_index]
        m_win = m[start_index:end_index]

        X_lst.append(X_win)
        y_lst.append(y_win[-1])  #Take the last value
        # y_lst.append(y_win)
        m_lst.append(m_win[-1])

        start_index += window_len - overlap_samples
        end_index += window_len - overlap_samples
    return np.array(X_lst), np.array(y_lst), np.array(m_lst)

def ang_to_vector(ang, unit = 'radians'):
    if unit == 'degrees':
        ang = ang /180 * np.pi
    x = ops.sin(ang)
    y = ops.cos(ang)
    return ops.stack([x,y], axis = 1)

def vector_to_ang(vector , unit = 'radians'):
    x = vector[:,0]
    y = vector[:,1]
    theta = ops.arctan2(x,y)
    if unit == 'degrees':
         return theta* 180 / np.pi
    elif unit == 'radians':
        return theta

def path_filter(pars, data_dir, mode):
    if type(pars) == dict:
        par_list = list(pars.values())
    elif type(pars) == list:
        par_list = pars

    file_list = [file_name for file_name in data_dir.iterdir() if file_name.stem[0] != "."]
    tar_list = file_list
    for par in par_list:
        if mode == 'include':
            tar_list = [e for e in tar_list if (par in str(e.stem))]
        elif mode == 'exclude':
            tar_list = [e for e in tar_list if (par not in str(e.stem))]
    return tar_list

if __name__ == "__main__":
    app()