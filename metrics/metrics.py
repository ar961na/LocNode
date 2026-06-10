"""ATE / RPE error metrics for trajectories."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_angles(quaternions_a, quaternions_b):
    """geodesic angle (rad) between paired rotations given as xyzw quaternions"""
    return (R.from_quat(quaternions_a).inv() * R.from_quat(quaternions_b)).magnitude()


def compute_ate(positions_gt, quaternions_gt, positions_a, quaternions_a):
    """
    compute absolute trajectory error (ATE) for positions and rotations

    Trajectories must be timestamp-associated (equal length, pose i of one
    corresponds to pose i of the other) and the algorithm trajectory should
    already be rigidly aligned to groundtruth (see align.umeyama_alignment).

    Parameters
    ----------
    positions_gt, quaternions_gt : numpy.ndarray
        parameters from groundtruth
    positions_a, quaternions_a : numpy.ndarray
        parameters from algorithm

    Returns
    -------
    ate_pos : float
        Root Mean Square Error (RMSE) for position ATE.
    ate_rot : float
        Mean rotational error in radians.
    """
    delta_pos = positions_gt - positions_a
    ate_pos = np.sqrt(np.mean(np.sum(delta_pos**2, axis=1)))
    ate_rot = np.mean(rotation_angles(quaternions_a, quaternions_gt))

    return ate_pos, ate_rot


def compute_cumulative_distances(positions):
    """compute cumulative distance of trajectory from positions"""
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.insert(np.cumsum(deltas), 0, 0.0)


def pairs_at_distance(cumulative_distances, distance, tolerance):
    """
    find index pairs (i, j) whose arc-length separation is
    distance +/- tolerance

    For each start index the closest matching end index is used, so each
    start contributes at most one pair. Fully vectorized: one searchsorted
    over the whole trajectory instead of a Python loop per start index.

    Returns
    -------
    starts, ends : (M,) numpy.ndarray of int
        paired indices into the trajectory
    """
    n = len(cumulative_distances)
    starts = np.arange(n)
    targets = cumulative_distances + distance

    # first index with arc length >= target; the best end index is either
    # this one or its left neighbour
    right = np.searchsorted(cumulative_distances, targets)
    left = right - 1

    def candidate_error(idx):
        # out-of-range or non-forward candidates get infinite error so they
        # are never selected
        err = np.full(n, np.inf)
        valid = (idx > starts) & (idx < n)
        err[valid] = np.abs(cumulative_distances[idx[valid]] - targets[valid])
        return err

    err_left = candidate_error(left)
    err_right = candidate_error(right)

    ends = np.where(err_left <= err_right, left, right)
    within = np.minimum(err_left, err_right) <= tolerance

    return starts[within], ends[within]


def compute_rpe(
    positions_gt, quaternions_gt, positions_a, quaternions_a, distances, tolerance=None
):
    """
    compute relative pose error (RPE) for positions and rotations

    For each pair of poses (i, j) separated by ~d meters along the
    groundtruth path, the relative motions
        D_gt = P_gt_i^-1 * P_gt_j,   D_a = P_a_i^-1 * P_a_j
    are compared via the error motion E = D_gt^-1 * D_a. RPE is invariant
    to the global frame, so no prior alignment is required.

    Trajectories must be timestamp-associated (equal length, pose i of one
    corresponds to pose i of the other). All pairs of a distance are
    evaluated in one batched scipy Rotation operation.

    Parameters
    ----------
    positions_gt, quaternions_gt : numpy.ndarray
        parameters from groundtruth
    positions_a, quaternions_a : numpy.ndarray
        parameters from algorithm
    distances : list of float
        sub-trajectory lengths in meters
    tolerance : float, optional
        max deviation from the requested length when selecting pose pairs
        (default: distance / 4)

    Returns
    -------
    rpe_pos : numpy.ndarray
        Mean translational RPE per distance (NaN if no pairs found).
    rpe_rot : numpy.ndarray
        Mean rotational RPE in radians per distance (NaN if no pairs found).
    """
    rotations_gt = R.from_quat(quaternions_gt)
    rotations_a = R.from_quat(quaternions_a)
    cumulative = compute_cumulative_distances(positions_gt)

    rpe_pos = np.full(len(distances), np.nan)
    rpe_rot = np.full(len(distances), np.nan)

    for n, d in enumerate(distances):
        tol = d / 4 if tolerance is None else tolerance
        i, j = pairs_at_distance(cumulative, d, tol)
        if len(i) == 0:
            continue

        # relative motions expressed in the local frame of pose i
        rot_gt_i_inv = rotations_gt[i].inv()
        rot_a_i_inv = rotations_a[i].inv()

        rel_pos_gt = rot_gt_i_inv.apply(positions_gt[j] - positions_gt[i])
        rel_pos_a = rot_a_i_inv.apply(positions_a[j] - positions_a[i])
        rpe_pos[n] = np.mean(np.linalg.norm(rel_pos_gt - rel_pos_a, axis=1))

        rel_rot_gt = rot_gt_i_inv * rotations_gt[j]
        rel_rot_a = rot_a_i_inv * rotations_a[j]
        rpe_rot[n] = np.mean((rel_rot_gt.inv() * rel_rot_a).magnitude())

    return rpe_pos, rpe_rot
