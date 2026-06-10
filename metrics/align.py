"""Trajectory association and alignment utilities."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def associate_timestamps(timestamps_gt, timestamps_a):
    """
    for each algorithm timestamp, find the index of the nearest
    groundtruth timestamp

    Parameters
    ----------
    timestamps_gt, timestamps_a : numpy.ndarray

    Returns
    -------
    indices : (N,) numpy.ndarray
        indices into timestamps_gt, one per algorithm timestamp
    """
    if len(timestamps_gt) == 0 or len(timestamps_a) == 0:
        raise ValueError("Timestamps cannot be empty.")
    if len(timestamps_gt) == 1:
        return np.zeros(len(timestamps_a), dtype=int)

    order = np.argsort(timestamps_gt)
    sorted_ts = timestamps_gt[order]

    pos = np.searchsorted(sorted_ts, timestamps_a)
    pos = np.clip(pos, 1, len(sorted_ts) - 1)
    take_left = np.abs(timestamps_a - sorted_ts[pos - 1]) <= np.abs(
        sorted_ts[pos] - timestamps_a
    )
    nearest = np.where(take_left, pos - 1, pos)

    return order[nearest]


def match_groundtruth_to_timestamps(
    timestamps_gt, positions_gt, quaternions_gt, timestamps_a
):
    """
    resample groundtruth trajectory at the algorithm timestamps
    (nearest-neighbour association)

    Parameters
    ----------
    timestamps_gt, positions_gt, quaternions_gt : numpy.ndarray
        parameters from groundtruth
    timestamps_a : numpy.ndarray
        timestamps from algorithm

    Returns
    -------
    matched_timestamps_gt, matched_positions_gt, matched_quaternions_gt : numpy.ndarray
        groundtruth parameters resampled at the algorithm timestamps
    """
    if len(positions_gt) != len(timestamps_gt):
        raise ValueError("Mismatch between timestamps and positions lengths.")

    idx = associate_timestamps(timestamps_gt, timestamps_a)
    return timestamps_gt[idx], positions_gt[idx], quaternions_gt[idx]


def umeyama_alignment(positions_gt, positions_a):
    """
    least-squares rigid transform (rotation + translation, no scale)
    mapping positions_a onto positions_gt (Umeyama / Horn method)

    Parameters
    ----------
    positions_gt, positions_a : (N, 3) numpy.ndarray
        paired positions (same length, same timestamps)

    Returns
    -------
    delta_R : (3, 3) numpy.ndarray
        rotation matrix
    delta_t : (3,) numpy.ndarray
        translation vector
    """
    mu_gt = positions_gt.mean(axis=0)
    mu_a = positions_a.mean(axis=0)

    # cross-covariance matrix
    C = (positions_gt - mu_gt).T @ (positions_a - mu_a) / len(positions_gt)
    U, _, Vt = np.linalg.svd(C)

    # handle reflection case
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0

    delta_R = U @ W @ Vt
    delta_t = mu_gt - delta_R @ mu_a

    return delta_R, delta_t


def apply_alignment(positions, quaternions, delta_R, delta_t):
    """
    apply a rigid transform (from umeyama_alignment) to a trajectory

    Parameters
    ----------
    positions : (N, 3) numpy.ndarray
    quaternions : (N, 4) numpy.ndarray, xyzw order
    delta_R : (3, 3) numpy.ndarray
    delta_t : (3,) numpy.ndarray

    Returns
    -------
    aligned_positions, aligned_quaternions : numpy.ndarray
    """
    aligned_positions = positions @ delta_R.T + delta_t
    aligned_quaternions = (R.from_matrix(delta_R) * R.from_quat(quaternions)).as_quat()

    return aligned_positions, aligned_quaternions
