#!/usr/bin/env python3
"""
Downsample point cloud or mesh file using voxel grid.

Supports input formats: .ply, .pcd, .obj (sampled from mesh).
Outputs downsampled cloud to specified path.
"""

import argparse
import open3d as o3d
import os


def main():
    """Parse arguments and downsample point cloud."""
    parser = argparse.ArgumentParser(description="Downsample point cloud or mesh.")
    parser.add_argument("--input", required=True, help="Input file (.ply, .pcd, .obj)")
    parser.add_argument("--output", required=True, help="Output downsampled file")
    parser.add_argument(
        "--voxel", type=float, default=0.1, help="Voxel size for downsampling"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000000,
        help="Number of points to sample from mesh (default: 1M)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    ext = os.path.splitext(args.input)[1].lower()

    if ext == ".obj":
        # Load mesh and sample points uniformly
        mesh = o3d.io.read_triangle_mesh(args.input)
        if not mesh.has_vertices():
            raise ValueError("Mesh has no vertices")
        pcd = mesh.sample_points_uniformly(number_of_points=args.num_points)
        print(f"Sampled {len(pcd.points)} points from mesh")
    else:
        # Load point cloud (.ply, .pcd, etc.)
        pcd = o3d.io.read_point_cloud(args.input)
        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty")
        print(f"Original points: {len(pcd.points)}")

    # Downsample using voxel grid
    down = pcd.voxel_down_sample(args.voxel)
    print(f"Downsampled points: {len(down.points)}")

    # Save result
    o3d.io.write_point_cloud(args.output, down)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
