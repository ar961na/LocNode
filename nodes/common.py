"""
Common utilities for point cloud processing and alignment.

Provides enums for alignment methods, point cloud processing helpers with feature caching,
and transformation utilities for trajectories.
"""

import os

os.environ["OPEN3D_HEADLESS_RENDERING"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from enum import Enum
import open3d as o3d
import numpy as np
from typing import Optional, Tuple
import hashlib
import pickle
from scipy.spatial.transform import Rotation

try:
    import rospy
except ImportError:
    rospy = None


class FastLIOState(Enum):
    """FAST-LIO mapping states."""

    IDLE = 0
    MAPPING = 1
    PUBLISHING = 2
    ERROR = 3


class GlobalAlignMethod(Enum):
    """Available global alignment methods."""

    FPFH_RANSAC = 0


class LocalAlignMethod(Enum):
    """Available local alignment methods."""

    P2P_ICP = 0  # Point-to-point ICP
    P2PLANE_ICP = 1  # Point-to-plane ICP (more robust)


class PointCloudProcessor:
    """Utilities for point cloud processing and feature computation with caching."""

    _cache_dir = None
    _use_cache = True
    # Memoized (load -> downsample -> normals) clouds, keyed by source params.
    # Holds static inputs like the reference cloud, so it is processed once
    # across many alignment cycles rather than per call.
    _cloud_cache = {}

    @staticmethod
    def set_cache_dir(cache_dir: str = None, use_cache: bool = True):
        """
        Enable/disable FPFH feature caching.

        Args:
            cache_dir: Directory for cached features (e.g., 'results/feature_cache').
                      If None, caching disabled.
            use_cache: Enable/disable caching (default: True).
        """
        PointCloudProcessor._cache_dir = cache_dir
        PointCloudProcessor._use_cache = use_cache

        if cache_dir and use_cache:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _get_cloud_hash(pcd: o3d.geometry.PointCloud, radius: float) -> str:
        """Generate unique hash for point cloud + radius combination."""
        points_hash = hashlib.md5(
            pcd.get_center().tobytes() + np.array([len(pcd.points), radius]).tobytes()
        ).hexdigest()
        return points_hash

    @staticmethod
    def _get_cache_path(pcd: o3d.geometry.PointCloud, radius: float) -> str:
        """Get cache file path for FPFH features."""
        if not PointCloudProcessor._cache_dir:
            return None
        cloud_hash = PointCloudProcessor._get_cloud_hash(pcd, radius)
        return os.path.join(
            PointCloudProcessor._cache_dir, f"fpfh_{cloud_hash}_{radius:.6f}.pkl"
        )

    @staticmethod
    def _load_cached_features(
        cache_path: str,
    ) -> Optional[o3d.pipelines.registration.Feature]:
        """Load cached FPFH features from disk."""
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [⚠] Cache load failed: {e}")
            return None

    @staticmethod
    def _save_cached_features(
        features: o3d.pipelines.registration.Feature, cache_path: str
    ) -> None:
        """Save FPFH features to disk cache."""
        if not cache_path:
            return
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)
        except Exception as e:
            print(f"  [⚠] Cache save failed: {e}")

    @staticmethod
    def load_pcd(filepath: str) -> o3d.geometry.PointCloud:
        """Load point cloud from PCD file."""
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {filepath}")
        return pcd

    @staticmethod
    def load_reference_cloud(filepath: str) -> o3d.geometry.PointCloud:
        """Load reference cloud from PCD, PLY, or OBJ format."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reference cloud not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pcd" or ext == ".ply":
            return PointCloudProcessor.load_pcd(filepath)
        elif ext == ".obj":
            mesh = o3d.io.read_triangle_mesh(filepath)
            pcd = mesh.sample_points_uniformly(number_of_points=int(1e6))
            if len(pcd.points) == 0:
                raise ValueError(f"Empty mesh/point cloud: {filepath}")
            return pcd
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .pcd, .obj, or .ply")

    @staticmethod
    def downsample(
        pcd: o3d.geometry.PointCloud, voxel_size: float
    ) -> o3d.geometry.PointCloud:
        """Downsample point cloud using voxel grid."""
        return pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def estimate_normals(
        pcd: o3d.geometry.PointCloud, radius: float = 0.1, max_nn: int = 30
    ) -> None:
        """Estimate surface normals using KDTree search."""
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )

    @staticmethod
    def load_downsampled(
        filepath: str,
        voxel_size: float,
        normal_radius: float = None,
        max_nn: int = 30,
    ) -> o3d.geometry.PointCloud:
        """
        Load a cloud from disk, downsample it, and optionally estimate normals.

        The expensive work (loading the full cloud, voxel downsampling, normal
        estimation) is memoized by (filepath, mtime, voxel_size, normal_radius,
        max_nn), so a static cloud such as the alignment reference is processed
        only once across many cycles; a modified file (newer mtime) invalidates
        its entry. A fresh copy is returned each call, so callers may safely
        mutate the result without corrupting the shared cache.
        """
        key = (
            filepath,
            os.path.getmtime(filepath),
            voxel_size,
            normal_radius,
            max_nn,
        )
        cached = PointCloudProcessor._cloud_cache.get(key)
        if cached is None:
            cached = PointCloudProcessor.downsample(
                PointCloudProcessor.load_pcd(filepath), voxel_size
            )
            if normal_radius is not None:
                PointCloudProcessor.estimate_normals(cached, normal_radius, max_nn)
            PointCloudProcessor._cloud_cache[key] = cached

        return o3d.geometry.PointCloud(cached)

    @staticmethod
    def compute_fpfh(
        pcd: o3d.geometry.PointCloud, radius: float = 0.5
    ) -> o3d.pipelines.registration.Feature:
        """
        Compute FPFH features with optional disk caching.

        Features are cached to avoid recomputation. Subsequent calls with same point cloud
        and radius load from cache (10-20% speedup).

        Args:
            pcd: Point cloud with normals estimated.
            radius: Search radius for feature computation.

        Returns:
            FPFH feature descriptor.
        """
        cache_path = (
            PointCloudProcessor._get_cache_path(pcd, radius)
            if PointCloudProcessor._use_cache
            else None
        )
        if cache_path:
            cached_features = PointCloudProcessor._load_cached_features(cache_path)
            if cached_features is not None:
                print(f"  [✓] Loaded FPFH from cache")
                return cached_features

        features = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )

        PointCloudProcessor._save_cached_features(features, cache_path)
        return features


def save_transform(transform: np.ndarray, filepath: str) -> None:
    """Save 4x4 transformation matrix to text file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savetxt(filepath, transform, fmt="%.8f")


def load_transform(filepath: str) -> np.ndarray:
    """Load 4x4 transformation matrix from text file."""
    trans = np.loadtxt(filepath)
    if trans.shape != (4, 4):
        raise ValueError(f"Transform shape must be (4,4), got {trans.shape}")
    return trans


def save_trajectory(trajectory: list, filepath: str) -> None:
    """Save trajectory to TUM format file (timestamp tx ty tz qx qy qz qw)."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        for timestamp, x, y, z, qx, qy, qz, qw in trajectory:
            f.write(
                f"{timestamp:.9f} {x:.8f} {y:.8f} {z:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
            )


def load_trajectory(filepath: str) -> list:
    """Load trajectory from TUM format file."""
    trajectory = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 8:
                    trajectory.append([float(x) for x in parts])
    return trajectory


def transform_delta(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """
    Difference between two 4x4 transforms as (rotation_deg, translation_m):
    geodesic angle of R_b @ R_a.T and distance between origins. Distinguishes
    basin switches from per-cycle ICP jitter.
    """
    R_delta = T_b[:3, :3] @ T_a[:3, :3].T
    rot_deg = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(R_delta).as_rotvec()))
    )
    trans_m = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
    return rot_deg, trans_m


def is_basin_switch(
    T_prev: np.ndarray, T_new: np.ndarray, rot_deg: float, trans_m: float
) -> Optional[Tuple[float, float]]:
    """
    Shared basin-switch policy (live node + offline rebuild): a transform that
    jumps from its predecessor by more than either threshold has left the
    previous basin. Returns the (rotation_deg, translation_m) delta, else None.
    """
    d_rot, d_trans = transform_delta(T_prev, T_new)
    if d_rot > rot_deg or d_trans > trans_m:
        return d_rot, d_trans
    return None


def transform_point(point: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to a 3D point."""
    p_homo = np.append(point, 1.0)
    p_transformed = T @ p_homo
    return p_transformed[:3]


def transform_trajectory(trajectory: list, T: np.ndarray) -> list:
    """
    Apply 4x4 transformation to trajectory (position + orientation).

    Args:
        trajectory: List of [timestamp, x, y, z, qx, qy, qz, qw].
        T: 4x4 transformation matrix.

    Returns:
        Transformed trajectory with same format.
    """
    if not trajectory:
        return []

    arr = np.array(trajectory)  # (N, 8)
    timestamps = arr[:, 0:1]  # (N, 1)

    T_odom = np.zeros((len(arr), 4, 4))  # (N, 4, 4)
    T_odom[:, :3, :3] = Rotation.from_quat(arr[:, 4:8]).as_matrix()  # (N, 3, 3)
    T_odom[:, :3, 3] = arr[:, 1:4]  # (N, 3)
    T_odom[:, 3, 3] = 1.0  # (N, )

    T_ref = np.einsum("ij,njk->nik", T, T_odom)  # (N, 4, 4)

    return np.column_stack(
        [timestamps, T_ref[:, :3, 3], Rotation.from_matrix(T_ref[:, :3, :3]).as_quat()]
    ).tolist()
