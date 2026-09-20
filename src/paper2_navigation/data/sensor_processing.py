"""Pure transformations for recorded VLP, CNC, and Vive measurements.

Hardware communication, serial access, logging, and robot control are
deliberately excluded from this analysis module.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def fft_window(samples):
    signal = np.asarray(samples, dtype=float)
    signal = signal - np.mean(signal)
    signal = signal / np.max(signal)
    signal = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.fft(signal)[: len(signal) // 2])
    frequencies = np.linspace(0, 25200, len(spectrum))
    return spectrum, frequencies


def calculate_peaks(spectrum, width=8):
    spectrum = np.asarray(spectrum)
    frequencies = np.linspace(0, 25000, len(spectrum))
    intensities = []
    for centre_frequency in (15000, 17000, 20000, 23000):
        centre_index = np.count_nonzero(frequencies < centre_frequency)
        lower = centre_index - width
        upper = centre_index + width
        index = spectrum[lower:upper].argmax() + lower
        intensities.append(spectrum[index])
    return intensities


def process_vlp(frame):
    """Extract four LED intensities from recorded ADC sample strings."""
    frame = frame.copy()
    frame.loc[:, "vlp_data"] = frame["vlp_data"].apply(
        lambda value: np.array(
            [
                reading
                for reading in eval(
                    value, {"__builtins__": {}, "array": np.array, "np": np}
                )
                if reading is not None
            ]
        )
    )
    frame.loc[:, "pks"] = frame["vlp_data"].apply(
        lambda readings: [
            np.array(calculate_peaks(fft_window(reading)[0])) for reading in readings
        ]
    )
    frame.loc[:, ["L1", "L2", "L3", "L4"]] = np.array(
        frame["pks"].apply(lambda peaks: np.mean(np.asarray(peaks), axis=0)).to_list()
    )
    frame.loc[:, ["L1_test", "L2_test", "L3_test", "L4_test"]] = np.array(
        frame["pks"].apply(lambda peaks: np.median(np.asarray(peaks[:1]), axis=0)).to_list()
    )
    return frame


def process_cnc(frame):
    frame = frame.copy()
    frame[["cnc_y", "cnc_x"]] = frame["cnc_data"].str.split(",", expand=True).astype(float)
    return frame


def parse_vive(frame):
    """Parse recorded Vive matrices and average repeated readings."""
    frame = frame.copy()

    def parse(value):
        return np.array(
            eval(
                value.replace("array", "np.array"),
                {"__builtins__": {}, "np": np, "object": object},
            )
        )

    def valid_matrices(values):
        return [matrix for matrix in values if matrix.tolist() is not None]

    def average_matrix(values):
        return np.mean([value["m"] for value in values], axis=0)

    frame["vive_data"] = frame["vive_data"].apply(parse).apply(valid_matrices)
    valid = frame["vive_data"].notna()
    frame.loc[valid, "vive_data"] = frame.loc[valid, "vive_data"].apply(average_matrix)
    return frame


class ViveToRobotTransform:
    """Derive and apply the recorded Vive-to-robot coordinate transform."""

    def __init__(self):
        self.transformation_matrix = None

    def derive_transform(self, frame):
        calibration = frame[frame["last_cmd"].str.startswith("CAL")].head(3)
        if len(calibration) != 3:
            raise ValueError("Three calibration points are required")
        robot_poses = np.array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 0.9], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ],
            dtype=float,
        )
        vive_poses = np.array(calibration["vive_data"].apply(np.asarray).to_list())
        self.transformation_matrix = self.align_poses(vive_poses, robot_poses)

    @staticmethod
    def align_poses(vive_poses, robot_poses):
        vive_centroid = np.mean(vive_poses[:, :3, 3], axis=0)
        robot_centroid = np.mean(robot_poses[:, :3, 3], axis=0)
        vive_stack = (vive_poses[:, :3, 3] - vive_centroid).T
        robot_stack = (robot_poses[:, :3, 3] - robot_centroid).T
        left, _, right_transpose = np.linalg.svd(vive_stack @ robot_stack.T)
        rotation = left @ right_transpose
        if np.linalg.det(rotation) < 0:
            right_transpose[-1, :] *= -1
            rotation = left @ right_transpose
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = robot_centroid - rotation @ vive_centroid
        return transformation

    def transform_pose(self, pose):
        if self.transformation_matrix is None:
            raise ValueError("derive_transform must be called first")
        return self.transformation_matrix @ pose


def add_bottom_row(matrix):
    return np.vstack((matrix, np.array([[0, 0, 0, 1]], dtype=matrix.dtype)))


def extract_pose_y_up(pose):
    x, y, z = pose[:3, 3]
    yaw, pitch, roll = Rotation.from_matrix(pose[:3, :3]).as_euler("zyx", degrees=True)
    return [x, y, z, yaw, pitch, roll]


def transform_vive_frame(frame):
    frame = frame.copy()
    transformer = ViveToRobotTransform()
    transformer.derive_transform(frame)
    valid = frame["vive_data"].notna()
    frame.loc[valid, "transformed_vive"] = frame.loc[valid, "vive_data"].apply(
        lambda value: transformer.transform_pose(add_bottom_row(value))
    )
    labels = ["vive_x", "vive_y", "vive_z", "vive_yaw", "vive_pitch", "vive_roll"]
    frame.loc[valid, labels] = np.array(
        frame.loc[valid, "transformed_vive"].apply(extract_pose_y_up).to_list()
    )
    return frame


def normalize_angle(angle):
    """Wrap angle(s), in degrees, to the interval [-180, 180]."""
    values = np.asarray(angle)
    wrapped = values % 360
    wrapped = np.where(wrapped > 180, wrapped - 360, wrapped)
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped.tolist() if isinstance(angle, list) else wrapped
