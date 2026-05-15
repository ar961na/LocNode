"""
Local point cloud alignment using ICP.

Fine alignment of point clouds using Iterative Closest Point (ICP) method.
Requires a good initial transformation and works best with overlapping clouds.
"""

import open3d as o3d
import numpy as np
from typing import Optional
from nodes.common import LocalAlignMethod, PointCloudProcessor


class LocalAlign:
    """ICP-based fine point cloud alignment."""

    def __init__(
        self,
        voxel_size: float = 0.1,
        threshold: float = 0.3,
        method: LocalAlignMethod = LocalAlignMethod.P2PLANE_ICP,
        use_robust: bool = True,
    ):
        """
        Initialize local aligner.

        Args:
            voxel_size: Downsampling voxel size.
            threshold: Distance threshold for ICP correspondence (meters).
            method: ICP variant (P2P_ICP or P2PLANE_ICP). P2PLANE is more robust.
            use_robust: Enable robust variants (not currently used).
        """
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.method = method
        self.use_robust = use_robust
        self.transformation = None
        self.fitness = None
        self.inlier_rmse = None

    def align(
        self, source_file: str, target_file: str, trans_init: np.ndarray
    ) -> np.ndarray:
        """
        Align source to target using initial transformation.

        Args:
            source_file: Path to source PCD file.
            target_file: Path to target PCD file.
            trans_init: Initial 4x4 transformation estimate.

        Returns:
            Refined 4x4 transformation matrix.
        """
        source = PointCloudProcessor.load_pcd(source_file)
        target = PointCloudProcessor.load_pcd(target_file)

        source_down = PointCloudProcessor.downsample(source, self.voxel_size)
        target_down = PointCloudProcessor.downsample(target, self.voxel_size)

        # Estimate normals for point-to-plane ICP
        radius_normal = self.voxel_size * 3
        PointCloudProcessor.estimate_normals(source_down, radius_normal, max_nn=30)
        PointCloudProcessor.estimate_normals(target_down, radius_normal, max_nn=30)

        if self.method == LocalAlignMethod.P2P_ICP:
            return self._p2p_icp_align(source_down, target_down, trans_init)
        elif self.method == LocalAlignMethod.P2PLANE_ICP:
            return self._p2plane_icp_align(source_down, target_down, trans_init)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")

    def _p2p_icp_align(self, source, target, trans_init):
        """Point-to-point ICP alignment."""
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
        )
        result = o3d.pipelines.registration.registration_icp(
            source, target, self.threshold, trans_init, estimation, criteria
        )
        self._store_result(result)
        return self.transformation

    def _p2plane_icp_align(self, source, target, trans_init):
        """Point-to-plane ICP alignment (more robust than point-to-point)."""
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
        )
        result = o3d.pipelines.registration.registration_icp(
            source, target, self.threshold, trans_init, estimation, criteria
        )
        self._store_result(result)
        return self.transformation

    def _store_result(self, result: o3d.pipelines.registration.RegistrationResult):
        """Store alignment results."""
        self.transformation = result.transformation
        self.fitness = result.fitness
        self.inlier_rmse = result.inlier_rmse

    def set_method(self, method: LocalAlignMethod):
        """Set ICP method (P2P or P2PLANE)."""
        self.method = method

    def set_voxel_size(self, voxel_size: float):
        """Set downsampling voxel size."""
        self.voxel_size = voxel_size

    def set_threshold(self, threshold: float):
        """Set ICP distance threshold."""
        self.threshold = threshold

    def get_fitness(self) -> Optional[float]:
        """Return fitness score from last alignment."""
        return self.fitness

    def get_inlier_rmse(self) -> Optional[float]:
        """Return RMSE of inlier correspondences."""
        return self.inlier_rmse
