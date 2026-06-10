"""Trajectory loading utilities."""

import numpy as np
import pandas as pd


def load_trajectory(filepath, quat_order="xyzw"):
    """
    load a trajectory from a TUM-format .txt file or a .csv file

    Each row must contain: timestamp x y z + quaternion (4 values).

    Parameters
    ----------
    filepath : str
        path to a .txt (TUM) or .csv trajectory file
    quat_order : {"xyzw", "wxyz"}
        quaternion component order in the file; returned quaternions are
        always in scipy order (qx, qy, qz, qw)

    Returns
    -------
    timestamps : (N,) numpy.ndarray
    positions : (N, 3) numpy.ndarray
    quaternions : (N, 4) numpy.ndarray, in (qx, qy, qz, qw) order

    Raises
    ------
    ValueError
        if filepath is not .csv or .txt format, or rows are malformed
    """
    if filepath.endswith(".csv"):
        data = pd.read_csv(filepath).to_numpy(dtype=float)
    elif filepath.endswith(".txt"):
        data = np.loadtxt(filepath, comments="#", dtype=float)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    data = np.atleast_2d(data)
    if data.shape[1] < 8:
        raise ValueError(
            f"Expected 8 columns (ts x y z + quaternion), got {data.shape[1]}"
        )

    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]

    if quat_order == "wxyz":
        quaternions = quaternions[:, [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"Unknown quaternion order: {quat_order}")

    # normalize each quaternion individually
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

    return timestamps, positions, quaternions


def save_trajectory(filepath, timestamps, positions, quaternions):
    """
    write a trajectory to a TUM-format .txt file (timestamp x y z qx qy qz qw)

    Quaternions are written in scipy xyzw order, so a file saved here reloads
    with the default ``quat_order="xyzw"``. Used to persist the time-cropped
    trajectories produced by analysis.py.

    Parameters
    ----------
    filepath : str
        destination path (.txt)
    timestamps : (N,) numpy.ndarray
    positions : (N, 3) numpy.ndarray
    quaternions : (N, 4) numpy.ndarray, xyzw order
    """
    data = np.column_stack([timestamps, positions, quaternions])
    np.savetxt(filepath, data, fmt=["%.9f"] + ["%.8f"] * 7)
