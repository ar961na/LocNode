#!/usr/bin/env python3
"""
Unified visualization script for point clouds and trajectories.

Supports:
- Point cloud visualization (PCD, PLY, OBJ formats)
- Trajectory visualization (TUM format with multiple trajectories)
- Reference cloud overlay
- GIF export with rotating camera view
- Interactive display

Usage examples:
    # Visualize point cloud as GIF
    python visualize.py --cloud cloud.ply --output animation.gif --fps 15 --duration 5

    # Visualize trajectory with reference cloud
    python visualize.py --traj trajectory.txt --reference cloud.ply --output traj.gif

    # Visualize multiple trajectories
    python visualize.py --traj traj1.txt traj2.txt --reference cloud.ply --visualize

    # Interactive visualization
    python visualize.py --cloud cloud.ply --visualize
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import tempfile
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from PIL import Image

# ============================================================================
# Point Cloud Loading
# ============================================================================


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from PCD, PLY, or OBJ format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pcd":
        pcd = o3d.io.read_point_cloud(filepath)
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(filepath)
    elif ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(filepath)
        pcd = mesh.sample_points_uniformly(number_of_points=int(1e6))
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .pcd, .ply, or .obj")

    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {filepath}")

    return pcd


# ============================================================================
# Trajectory Loading and Conversion
# ============================================================================


def load_trajectory(filepath: str) -> list:
    """Load trajectory from TUM format file (timestamp x y z qx qy qz qw)."""
    trajectory = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 8:
                    trajectory.append([float(x) for x in parts])
    return trajectory


def trajectory_to_poses(trajectory: list) -> list:
    """Convert trajectory to list of 4x4 pose matrices."""
    poses = []
    for timestamp, x, y, z, qx, qy, qz, qw in trajectory:
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        R = Rotation.from_quat([qx, qy, qz, qw])
        T[:3, :3] = R.as_matrix()
        poses.append(T)
    return poses


def create_trajectory_visualization(
    trajectory: list, color: list, scale: float = 0.02
) -> list:
    """Create 3D visualization geometries (boxes + lines) from trajectory."""
    geometries = []
    poses = trajectory_to_poses(trajectory)

    # Downsample to every 5th pose for clarity
    for i, pose in enumerate(poses[::5]):
        if i == 0 or i == len(poses[::5]) - 1:
            size = scale * 2  # Start and end larger
        else:
            size = scale

        # Create small box for each pose
        box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        box.paint_uniform_color(color)
        box.transform(pose)
        geometries.append(box)

        # Connect poses with lines
        if i > 0:
            prev_pose = poses[::5][i - 1]
            line_points = np.array([prev_pose[:3, 3], pose[:3, 3]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color(color)
            geometries.append(line_set)

    return geometries


# ============================================================================
# GIF Generation
# ============================================================================


def save_rotating_gif(
    geometries: list,
    output_path: str,
    fps: int = 10,
    duration: int = 5,
    width: int = 1024,
    height: int = 768,
    center: np.ndarray = None,
    extent: float = None,
):
    """
    Save a rotating GIF of geometries using rotating camera.
    Uses Z-up coordinate system.

    Args:
        geometries: List of Open3D geometries to visualize
        output_path: Output GIF file path
        fps: Frames per second
        duration: Duration of full rotation in seconds
        width: Output image width
        height: Output image height
        center: Optional center point for camera lookat
        extent: Optional extent for zoom calculation
    """
    print(f"\nGenerating rotating GIF: {output_path}")

    # Compute center and extent from geometries if not provided
    if center is None or extent is None:
        points = []
        for geom in geometries:
            try:
                if hasattr(geom, "points"):
                    pts = np.asarray(geom.points)
                    if len(pts) > 0:
                        points.extend(pts)
                elif hasattr(geom, "vertices"):
                    verts = np.asarray(geom.vertices)
                    if len(verts) > 0:
                        points.extend(verts)
            except:
                pass

        if points:
            points = np.array(points)
            if center is None:
                center = np.mean(points, axis=0)
            if extent is None:
                extent = np.max(np.linalg.norm(points - center, axis=1))
        else:
            center = np.array([0.0, 0.0, 0.0])
            extent = 1.0

    print(f"  Center: {center}")
    print(f"  Extent: {extent:.3f}")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for geom in geometries:
        vis.add_geometry(geom)

    # Setup camera view (Z-up coordinate system)
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_zoom(0.8 / extent if extent > 0 else 0.8)
    ctr.set_front([1, 0, 0])  # Looking along +X initially
    ctr.set_up([0, 0, 1])  # Z is up

    vis.poll_events()
    vis.update_renderer()

    # Capture frames with rotating camera
    num_frames = fps * duration
    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        for frame_idx in range(num_frames):
            # Rotate around the vertical (Z) axis
            angle_increment = (360.0 / num_frames) * 5.8
            ctr.rotate(angle_increment, 0)

            vis.poll_events()
            vis.update_renderer()

            # Capture frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            vis.capture_screen_image(frame_path, do_render=True)

            try:
                frame_image = Image.open(frame_path)
                frames.append(frame_image)
            except Exception as e:
                print(f"  Warning: Failed to load frame {frame_idx}: {e}")

            if (frame_idx + 1) % 10 == 0:
                print(f"  Captured {frame_idx + 1}/{num_frames} frames")

        # Save GIF
        if frames:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            print(f"  GIF saved: {output_path}")
        else:
            print("  Error: No frames captured")

    finally:
        vis.destroy_window()
        shutil.rmtree(temp_dir)


# ============================================================================
# Main Visualization Logic
# ============================================================================


def visualize_point_cloud(
    cloud_path: str,
    reference_path: str = None,
    output_gif: str = None,
    fps: int = 10,
    duration: int = 5,
    width: int = 1024,
    height: int = 768,
    visualize: bool = False,
):
    """Visualize a single point cloud with optional reference cloud."""
    print("Loading point cloud...")
    pcd = load_point_cloud(cloud_path)
    print(f"  Loaded {len(pcd.points)} points")

    geometries = [pcd]

    # Load reference cloud if provided
    if reference_path and os.path.exists(reference_path):
        print("Loading reference cloud...")
        ref_cloud = load_point_cloud(reference_path)
        ref_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
        geometries.append(ref_cloud)

    # Get center and extent for camera positioning
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_extent()

    # Save as GIF or display
    if output_gif:
        save_rotating_gif(
            geometries,
            output_gif,
            fps=fps,
            duration=duration,
            width=width,
            height=height,
            center=center,
            extent=extent,
        )
    elif visualize:
        print("\nDisplaying... (close window to exit)")
        o3d.visualization.draw_geometries(geometries)
    else:
        print("Use --output to save as GIF or --visualize to display interactively")


def visualize_trajectories(
    traj_files: list,
    reference_path: str = None,
    output_gif: str = None,
    fps: int = 10,
    duration: int = 1,
    width: int = 1024,
    height: int = 768,
    visualize: bool = False,
):
    """Visualize one or more trajectories with optional reference cloud."""
    print("Loading trajectories...")

    trajectories = []
    colors = [
        [0.0, 0.8, 0.0],  # Green
        [0.8, 0.0, 0.0],  # Red
        [0.0, 0.0, 0.8],  # Blue
        [0.8, 0.8, 0.0],  # Yellow
        [0.8, 0.0, 0.8],  # Magenta
        [0.0, 0.8, 0.8],  # Cyan
    ]

    # Load all trajectories
    for i, traj_file in enumerate(traj_files):
        traj = load_trajectory(traj_file)
        trajectories.append(traj)
        print(
            f"  Trajectory {i+1}: {len(traj)} poses from {os.path.basename(traj_file)}"
        )

    geometries = []

    # Create visualizations for each trajectory
    for i, (trajectory, traj_file) in enumerate(zip(trajectories, traj_files)):
        if trajectory:
            print(f"Creating visualization for {os.path.basename(traj_file)}...")
            color = colors[i % len(colors)]
            geoms = create_trajectory_visualization(trajectory, color)
            geometries.extend(geoms)

    # Add reference cloud if provided
    if reference_path and os.path.exists(reference_path):
        print("Loading reference cloud...")
        ref_cloud = load_point_cloud(reference_path)
        ref_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
        geometries.append(ref_cloud)

    # Save as GIF or display
    if output_gif:
        save_rotating_gif(
            geometries,
            output_gif,
            fps=fps,
            duration=duration,
            width=width,
            height=height,
        )
    elif visualize:
        print("\nVisualization legend:")
        for i in range(len(traj_files)):
            color_name = ["Green", "Red", "Blue", "Yellow", "Magenta", "Cyan"][i % 6]
            print(f"  {color_name}: Trajectory {i+1}")
        if reference_path:
            print("  Gray: Reference point cloud")
        print("\nDisplaying... (close window to exit)")
        o3d.visualization.draw_geometries(geometries)
    else:
        print("Use --output to save as GIF or --visualize to display interactively")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified visualization for point clouds and trajectories"
    )

    # Input options (one is required)
    input_group = parser.add_argument_group("Input (choose one)")
    input_group.add_argument(
        "--cloud", "-c", help="Point cloud file (.pcd, .ply, .obj)"
    )
    input_group.add_argument(
        "--traj", nargs="+", help="Trajectory file(s) in TUM format"
    )

    # Optional overlays
    parser.add_argument(
        "--reference", "-r", help="Reference point cloud (.pcd, .ply, .obj)"
    )

    # Output options
    output_group = parser.add_argument_group("Output (choose one)")
    output_group.add_argument("--output", "-o", help="Save as GIF to this path")
    output_group.add_argument(
        "--visualize", "-v", action="store_true", help="Show interactive visualization"
    )

    # GIF options
    gif_group = parser.add_argument_group("GIF options")
    gif_group.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    gif_group.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration of full rotation in seconds (default: 5)",
    )
    gif_group.add_argument(
        "--width", type=int, default=1024, help="Output image width (default: 1024)"
    )
    gif_group.add_argument(
        "--height", type=int, default=768, help="Output image height (default: 768)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.cloud and not args.traj:
        parser.error("Please provide either --cloud or --traj")
    if args.cloud and args.traj:
        parser.error("Please provide either --cloud or --traj, not both")

    # Default to visualize if no output specified
    if not args.output and not args.visualize:
        args.visualize = True

    # Handle point cloud visualization
    if args.cloud:
        visualize_point_cloud(
            args.cloud,
            reference_path=args.reference,
            output_gif=args.output,
            fps=args.fps,
            duration=args.duration,
            width=args.width,
            height=args.height,
            visualize=args.visualize,
        )
    # Handle trajectory visualization
    elif args.traj:
        visualize_trajectories(
            args.traj,
            reference_path=args.reference,
            output_gif=args.output,
            fps=args.fps,
            duration=args.duration,
            width=args.width,
            height=args.height,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
