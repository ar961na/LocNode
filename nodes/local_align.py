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
    ):
        """
        Initialize local aligner.

        Args:
            voxel_size: Downsampling voxel size.
            threshold: Distance threshold for ICP correspondence (meters).
            method: ICP variant (P2P_ICP or P2PLANE_ICP).
        """
        self.voxel_size = voxel_size
        self.threshold = threshold
        self.method = method
        self.transformation = None
        self.fitness = None
        self.inlier_rmse = None
        self.seed_fitness = None
        self.selected_seed = None
        self.diverged = False

    def align(
        self,
        source: o3d.geometry.PointCloud,
        target_file: str,
        trans_init: np.ndarray,
        candidate_seeds: list = None,
    ) -> np.ndarray:
        """
        Align source to target using initial transformation.

        When candidate_seeds are supplied, the seed with the highest fitness
        (scored on the same downsampled clouds and threshold ICP uses) is
        chosen before refinement. This lets the caller offer alternatives such
        as the previous cycle's transform alongside a fresh global estimate;
        ICP then warm-starts from whichever is genuinely closer.

        Args:
            source: Source point cloud object.
            target_file: Path to target PCD file (cached after first load).
            trans_init: Initial 4x4 transformation estimate.
            candidate_seeds: Optional list of additional 4x4 seeds to consider.

        Returns:
            Refined 4x4 transformation matrix.
        """
        radius_normal = self.voxel_size * 3

        source_down = PointCloudProcessor.downsample(source, self.voxel_size)
        PointCloudProcessor.estimate_normals(source_down, radius_normal, max_nn=30)

        target_down = PointCloudProcessor.load_downsampled(
            target_file, self.voxel_size, radius_normal, max_nn=30
        )

        # Pick the best-scoring seed on a consistent metric before refining.
        trans_init = self._select_seed(
            source_down, target_down, trans_init, candidate_seeds
        )

        if self.method == LocalAlignMethod.P2P_ICP:
            self._p2p_icp_align(source_down, target_down, trans_init)
        elif self.method == LocalAlignMethod.P2PLANE_ICP:
            self._p2plane_icp_align(source_down, target_down, trans_init)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")

        return self._resolve_against_seed()

    def _evaluate(self, source_down, target_down, transform) -> float:
        """Fitness of a transform on the given clouds at the ICP threshold (no iterations)."""
        return o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, self.threshold, transform
        ).fitness

    def _select_seed(self, source_down, target_down, trans_init, candidate_seeds):
        """
        Return the highest-fitness seed among trans_init and any candidates.

        All seeds are scored on the same downsampled clouds and threshold ICP
        uses, so this selection and the later keep/fallback decision in
        _resolve_against_seed are comparable on one consistent metric. Stores
        the chosen seed and its fitness for retrieval.
        """
        seeds = [trans_init]
        if candidate_seeds:
            seeds.extend(candidate_seeds)

        self.selected_seed = trans_init
        self.seed_fitness = -1.0
        for seed in seeds:
            fitness = self._evaluate(source_down, target_down, seed)
            if fitness > self.seed_fitness:
                self.selected_seed = seed
                self.seed_fitness = fitness

        return self.selected_seed

    def _resolve_against_seed(self) -> np.ndarray:
        """
        Apply the keep/fallback policy after ICP.

        Keep the ICP result unless it regressed against its seed; if it did
        (ICP diverged), fall back to the seed.
        """
        self.diverged = self.fitness < self.seed_fitness
        if self.diverged:
            self.transformation = self.selected_seed
            self.fitness = self.seed_fitness
        return self.transformation

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

    def get_seed_fitness(self) -> Optional[float]:
        """Return fitness of the seed chosen for the last alignment."""
        return self.seed_fitness

    def has_diverged(self) -> bool:
        """Return whether the last ICP regressed against its seed and fell back."""
        return self.diverged
