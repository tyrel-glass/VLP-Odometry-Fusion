import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import paper2_navigation.data.sensor_processing as c

import paper2_navigation.modeling.gen_cnc_vlp_model as vlp

from pathlib import Path

import typer
from loguru import logger

from paper2_navigation.config import INTERIM_DATA_DIR, RAW_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()



enc_per_cm =   39.816195443518865



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):

    files = [f'exp1_{i}' for i in range(0,10)]

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing experment dataset")

    input_file = RAW_DATA_DIR / 'navigation_training/' / files[0]

    vlp_models = vlp.load_vlp_models()


    output_file = INTERIM_DATA_DIR / 'navigation_training/' / files[0].split('.')[0] /'.pkl'

    for vlp_name, vlp_model in vlp_models.items():
        for filename in files:
            logger.info(f"processing {filename} with {vlp_name}")
            input_file = RAW_DATA_DIR / 'navigation_training/' / (filename+'.csv')
            output_filename = f'{filename}_{vlp_name}.csv'
            output_file = INTERIM_DATA_DIR / 'navigation_training/' /output_filename

            df = process_robot_exp_file(input_file, vlp_model)
            # pickle.dump(run_data_dic, open(output_file , 'wb'))
            df.to_csv(output_file )


    logger.success("Processing dataset complete.")
    # -----------------------------------------


def process_robot_exp_file(input_file, vlp_model):
    logger.success("opending file: ", input_file)
    df= pd.read_csv(input_file, delimiter = '|')
    df = c.parse_vive(df)
    df = c.transform_vive_frame(df)

    #======================== Drop the calibration rows ============================
    df = df[~df['last_cmd'].str.contains('CAL:')]#remove cal points
    #========================================================================


    # parse the move cmds before dropping the move rows
    def parse_last_turn(cmd):
        if 'TURN:' in cmd:
            return -int(float(cmd.split('TURN:')[1]))
        else:
            return np.nan

    df['encoder_heading_change_step'] = df['last_cmd'].apply(parse_last_turn)
    no_turn_filt =  df['last_cmd'].str.contains('MOVE:') & df['last_cmd'].shift(1).str.contains('MOVE:')
    df.loc[no_turn_filt, 'encoder_heading_change_step'] = 0
    df['encoder_heading_change_step'] = df['encoder_heading_change_step'].ffill()

# new test----------------
# df['encoder_heading_change'] = df['encoder_heading_change_step'] / enc_per_degree
# new test----------------
    def steps_to_ang(steps):
        acw_line = np.poly1d([ 0.18816057, -0.82087375])
        cw_line  = np.poly1d([0.18951681, 1.00878798])
        if steps > 0:
            ang = cw_line(steps)
        elif steps < 0:
            ang = acw_line(steps)
        else:
            ang = 0
        return ang
    df['encoder_heading_change'] = df['encoder_heading_change_step'].apply(steps_to_ang)
# new test end----------------


    #======================== Drop the move rows ============================
    #========================================================================
    df = df[df['last_cmd'].str.contains('MOVE')]  #remove turn datasteps
    df.reset_index(inplace = True)
    #========================================================================
    #========================================================================


    # extract rss values for each light
    df = c.process_vlp(df)
    # predict new vlp locations
    df[['vlp_x_hist', 'vlp_y_hist']] = vlp_model.predict(df[['L1', 'L2', 'L3', 'L4']].values)/1000 #cnc in mm

    # ======================================================================
    # ==============================  TESTING   =======================
    # ======================================================================
    # ======================================================================
    df[['vlp_x_hist_test', 'vlp_y_hist_test']] = vlp_model.predict(df[['L1_test', 'L2_test', 'L3_test', 'L4_test']].values)/1000 #cnc in mm
    # ======================================================================
    # ======================================================================



    # map between robot center and vive tracker
    df['x_hist'] = df['vive_x'] +0.07*np.sin(df['vive_yaw']/180*np.pi)
    df['y_hist'] = df['vive_z'] +0.07*np.cos(df['vive_yaw']/180*np.pi)

    # take heading from vive (offset by 180 degrees)
    df['heading_hist'] = [c.normalize_angle(a) for a in (df['vive_yaw'] + 180)]


        # ============================ CALCULATE HEADING CHANGE STATS========================
    df['vlp_heading_hist'] = np.arctan2(df['vlp_x_hist'].diff(1) , df['vlp_y_hist'].diff(1)) *180/np.pi

    df['vive_heading_change'] = df['heading_hist'].diff().apply(c.normalize_angle)
    df['vlp_heading_change'] = df['vlp_heading_hist'].diff().apply(c.normalize_angle)


    # ----------------- Compute errs  ----------------------------------
    df['encoder_heading_change_err'] = df['vive_heading_change'] - df['encoder_heading_change']
    df['vlp_heading_change_err'] = df['vive_heading_change'] - df['vlp_heading_change']

    #===========================================================================

    # ============================ CALCULATE STEP CHANGE STATS========================
    def parse_last_move(cmd):
        if 'MOVE:' in cmd:
            return int(float(cmd.split('MOVE:')[1]))
        else:
            return np.nan
    df['encoder_location_change_step'] = df['last_cmd'].apply(parse_last_move)
    df['encoder_location_change'] = df['encoder_location_change_step']/ enc_per_cm /100

    df['vive_location_change'] = np.sqrt(np.square(df['x_hist'].diff(1)) + np.square(df['y_hist'].diff(1)))

    df['vlp_location_change'] = np.sqrt(np.square(df['vlp_x_hist'].diff(1)) + np.square(df['vlp_y_hist'].diff(1)))

    df.loc[:,'encoder_location_change_err'] = df['vive_location_change'] - df['encoder_location_change']
    df['vlp_location_change_err'] = df['vive_location_change'] - df['vlp_location_change']


    df = calc_encoder_heading_hist(df)

    df = calc_encoder_xy_hist(df)
    df.reset_index(drop = True, inplace = True)

    #derive radian based angles
    df['heading_hist_rad'] = np.radians(df['heading_hist'])
    df['encoder_heading_hist_rad'] = np.radians(df['encoder_heading_hist'])
    df['encoder_heading_change_rad'] = np.radians(df['encoder_heading_change'])
    df['encoder_heading_change_err_rad'] = np.radians(df['encoder_heading_change_err'])
    df['vlp_heading_hist_rad'] = np.radians(df['vlp_heading_hist'])
    df['vlp_heading_change_rad'] = np.radians(df['vlp_heading_change'])

    return df


def convert_df_to_dic(df):
    X_data = np.array([
        # df['encoder_x_hist'].to_list(),
        # df['encoder_y_hist'].to_list(),
        # df['encoder_heading_hist'].to_list(),
        # df['encoder_location_change'].to_list(),
        # df['encoder_location_change'].to_list(),

        df['vlp_x_hist'].to_list(),
        df['vlp_y_hist'].to_list(),
        df['encoder_heading_change'].to_list(),
        # df['vlp_heading_hist'].to_list()
        ]).T

    y_data = np.array([
        df['x_hist'].to_list(),
        df['y_hist'].to_list(),
        df['heading_hist'].to_list()
        ]).T

    data_dic = {'X':X_data, 'y':y_data}
    return data_dic

# ------------------------------------------------ Functions to process VLP data -------------------------------------



from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D support is loaded


def plot_surface_irregular(df, cmap='cividis', grid_resolution=100):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig = plt.figure(figsize=(8.5, 6.2))
    target_peaks = ['L1', 'L2', 'L3', 'L4']

    # Grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(df["cnc_x"].min(), df["cnc_x"].max(), grid_resolution),
        np.linspace(df["cnc_y"].min(), df["cnc_y"].max(), grid_resolution)
    )

    # Use GridSpec for finer layout control
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.1)

    for i, peak in enumerate(target_peaks):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col], projection='3d')

        # Interpolate surface
        grid_z = griddata(
            points=(df["cnc_x"], df["cnc_y"]),
            values=df[peak],
            xi=(grid_x, grid_y),
            method='linear'
        )

        # Plot surface
        ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, linewidth=0, antialiased=True)

        # Labels and formatting
        ax.set_title(f'{peak} RSS Surface', pad=4)
        ax.set_xlabel('X (mm)', labelpad=2)
        ylab = ax.set_ylabel('Y (mm)', labelpad=2)
        ylab.set_rotation(90)
        zlab = ax.set_zlabel('RSS', labelpad=2)
        zlab.set_rotation(90)
    # Force manual layout to avoid clipping
    plt.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.08)


    fig.suptitle("Interpolated RSS Surfaces for LEDs L1–L4", fontsize=9)
    plt.tight_layout()
    return fig


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

def plot_rss_contours(df, cmap='cividis', grid_resolution=100):
    import matplotlib.gridspec as gridspec

    target_peaks = ['L1', 'L2', 'L3', 'L4']

    # Set up figure and a grid with extra column for colorbar
    fig = plt.figure(figsize=(3.5, 6))
    gs = gridspec.GridSpec(4, 2, width_ratios=[20, 1], wspace=0.05, hspace=0.4)

    contour_plot = None
    axs = []
    for i, peak in enumerate(target_peaks):
        ax = fig.add_subplot(gs[i, 0])
        axs.append(ax)

        # Grid for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(df["cnc_x"].min(), df["cnc_x"].max(), grid_resolution),
            np.linspace(df["cnc_y"].min(), df["cnc_y"].max(), grid_resolution)
        )
        grid_z = griddata(
            (df["cnc_x"], df["cnc_y"]),
            df[peak],
            (grid_x, grid_y),
            method='linear'
        )

        contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap=cmap)
        ax.set_title(f'{peak} RSS', fontsize=8, pad=2)
        if i == len(target_peaks) - 1:
            ax.set_xlabel('X (mm)', fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    # Colorbar placed in the rightmost column of the grid
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(contour_plot, cax=cax)
    cbar.set_label('RSS', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig

def calc_encoder_heading_hist(df):
    # -------------  Calculate  --------
    ang_deltas = (df['encoder_heading_change']).to_list()[1:] #values are to get to current step
    new_lst = []
    for i in range(len(ang_deltas)+1):
        new_lst.append(c.normalize_angle(np.sum(ang_deltas[:i])))

    df['encoder_heading_hist'] = np.array(new_lst) + df['heading_hist'].iloc[0] #map encoder to initial heading
    df['encoder_heading_hist'] = df['encoder_heading_hist'].apply(c.normalize_angle)

    return df

def calc_encoder_xy_hist(df):

    heading = df['encoder_heading_hist'].to_numpy()
    dx = np.sin(heading/180*np.pi) * df['encoder_location_change']
    dy = np.cos(heading/180*np.pi) * df['encoder_location_change']

    dx_sum = [sum(dx[1:i]) for i in range(1,len(dx)+1)]
    dy_sum = [sum(dy[1:i]) for i in range(1,len(dy)+1)]


    df['encoder_x_hist'] = np.array(dx_sum) + df['x_hist'].iloc[0]
    df['encoder_y_hist'] = np.array(dy_sum) + df['y_hist'].iloc[0]

    return df



def filter_outliers(data, threshold=1.5):

    data = np.asarray(data)  # Ensure input is a NumPy array
    data = data[~np.isnan(data)]
    q1 = np.percentile(data, 25)  # First quartile (Q1)
    q3 = np.percentile(data, 75)  # Third quartile (Q3)
    iqr = q3 - q1  # Interquartile range (IQR)

    # Calculate bounds for non-outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Filter the data
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

if __name__ == "__main__":
    app()
