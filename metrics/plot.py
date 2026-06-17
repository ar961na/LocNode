"""Optional Open3D plotting of aligned trajectories."""

import colorsys

import numpy as np
from scipy.spatial.transform import Rotation as R

import align


def trajectory_frames(positions, quaternions, color=(1, 0, 0), size=0.03):
    """
    build open3d coordinate-frame line sets for each pose

    Parameters
    ----------
    positions, quaternions : numpy.ndarray
    color : tuple, default=(1, 0, 0)
    size : float
        axis length of each frame

    Returns
    -------
    frames : list of open3d.geometry.LineSet
    """
    import open3d as o3d

    base_points = np.array([[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
    base_lines = np.array([[0, 1], [0, 2], [0, 3]])

    frames = []
    for pos, quat in zip(positions, quaternions):
        frame = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(base_points),
            lines=o3d.utility.Vector2iVector(base_lines),
        )
        frame.paint_uniform_color(color)
        frame.rotate(R.from_quat(quat).as_matrix(), center=(0, 0, 0))
        frame.translate(pos)
        frames.append(frame)

    return frames


def plot_trajectories(timestamps, positions, quaternions):
    """
    plot (single / multiple) trajectories from timestamps, positions and
    quaternions, resampled at the timestamps of the shortest trajectory

    Parameters
    ----------
    timestamps, positions, quaternions : lists of numpy.ndarray
    """
    import open3d as o3d

    colors = [
        colorsys.hsv_to_rgb(i / len(positions), 1, 1) for i in range(len(positions))
    ]

    # resample all trajectories at the timestamps of the shortest one
    idx_min = int(np.argmin([len(ts) for ts in timestamps]))

    frames = []
    for t, p, q, color in zip(timestamps, positions, quaternions, colors):
        _, p, q = align.match_groundtruth_to_timestamps(t, p, q, timestamps[idx_min])
        frames += trajectory_frames(p, q, color)

    o3d.visualization.draw_geometries(frames)
