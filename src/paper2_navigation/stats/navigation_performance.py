import numpy as np



def calc_loc_err(pre, true):
    x_d = pre[:,0] - true[:,0]
    y_d = pre[:,1] - true[:,1]
    errs = np.sqrt(np.square(x_d) + np.square(y_d))
    return errs


def normalize_angle_rad(angle):
    """Normalize an angle in radians to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi