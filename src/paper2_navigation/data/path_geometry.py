"""Geometry helpers used by analysis plots."""

import numpy as np


def create_polygon_targets(number, radius=2, center=(3.5, 3)):
    angles = np.linspace(0, 2 * np.pi, number + 1) + np.pi / 4
    return np.array(
        [
            [radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[1]]
            for angle in angles
        ]
    )
