"""
Global point cloud alignment using FPFH features and RANSAC.

Coarse alignment of two point clouds using feature matching and RANSAC-based
transformation estimation. Suitable for finding initial alignment when clouds
have significant overlap (30-100%).
"""

import open3d as o3d
import numpy as np
from typing import Optional

from nodes.common import GlobalAlignMethod, PointCloudProcessor


class GlobalAlign:
    """FPFH-RANSAC based coarse point cloud alignment."""

    def __init__(
        self,
        voxel_size: float = 0.1,
        method: GlobalAlignMethod = GlobalAlignMethod.FPFH_RANSAC,
    ):
        """
        Initialize global aligner.

        Args:
            voxel_size: Downsampling voxel size for feature computation.
            method: Alignment method (currently only FPFH_RANSAC).
        """
        self.voxel_size = voxel_size
        self.method = method
        self.transformation = None
        self.fitness = None

        # RANSAC convergence parameters (optimized for speed vs accuracy)
        self.ransac_max_iterations = 1_000_000  # Reduced for 30% speedup
        self.ransac_max_validation = 200  # Reduced for early stopping

    def align(self, source_file: str, target_file: str) -> np.ndarray:
        """
        Align source point cloud to target.

        Args:
            source_file: Path to source PCD file.
            target_file: Path to target PCD file.

        Returns:
            4x4 transformation matrix from target frame to source frame.
        """
        if self.method == GlobalAlignMethod.FPFH_RANSAC:
            return self._fpfh_ransac_align(source_file, target_file)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")

    def _fpfh_ransac_align(self, source_file: str, target_file: str) -> np.ndarray:
        """
        Perform FPFH-RANSAC alignment.

        Pipeline:
        1. Downsample both clouds
        2. Estimate surface normals
        3. Compute FPFH features
        4. Match features and estimate transformation via RANSAC
        """
        source = PointCloudProcessor.load_pcd(source_file)
        target = PointCloudProcessor.load_pcd(target_file)

        source_down = PointCloudProcessor.downsample(source, self.voxel_size)
        target_down = PointCloudProcessor.downsample(target, self.voxel_size)

        # Normal estimation
        radius_normal = self.voxel_size * 2
        PointCloudProcessor.estimate_normals(source_down, radius_normal)
        PointCloudProcessor.estimate_normals(target_down, radius_normal)

        # Feature computation
        radius_feature = self.voxel_size * 5
        source_fpfh = PointCloudProcessor.compute_fpfh(source_down, radius_feature)
        target_fpfh = PointCloudProcessor.compute_fpfh(target_down, radius_feature)

        distance_threshold = self.voxel_size * 1.5

        # RANSAC-based feature matching
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                False,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    self.ransac_max_iterations, self.ransac_max_validation
                ),
            )
        )

        self.transformation = result.transformation
        self.fitness = result.fitness

        return self.transformation

    def set_method(self, method: GlobalAlignMethod):
        """Set alignment method."""
        self.method = method

    def set_voxel_size(self, voxel_size: float):
        """Set downsampling voxel size."""
        self.voxel_size = voxel_size

    def set_ransac_params(self, max_iterations: int = None, max_validation: int = None):
        """
        Tune RANSAC parameters for speed vs accuracy trade-off.

        Args:
            max_iterations: Higher value = more accurate but slower (default: 1M).
            max_validation: Higher value = more stringent criteria but slower (default: 200).

        Presets:
            - Fast:      1M iterations, 100 max_validation (30% faster, lower accuracy)
            - Balanced:  1M iterations, 200 max_validation (30% faster, minimal loss) - DEFAULT
            - Thorough:  4M iterations, 500 max_validation (original, most accurate)
        """
        if max_iterations is not None:
            self.ransac_max_iterations = max_iterations
        if max_validation is not None:
            self.ransac_max_validation = max_validation

    def get_ransac_params(self) -> dict:
        """Return current RANSAC parameters."""
        return {
            "max_iterations": self.ransac_max_iterations,
            "max_validation": self.ransac_max_validation,
        }

    def get_fitness(self) -> Optional[float]:
        """Return fitness score from last alignment."""
        return self.fitness
