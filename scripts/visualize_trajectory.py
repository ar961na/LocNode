#!/usr/bin/env python3
"""
Visualize and compare trajectories with reference point cloud.

Supports multiple trajectories with different colors, optional reference cloud overlay,
and GIF export with rotating camera view.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import open3d as o3d
from PIL import Image
import tempfile
import shutil


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


def load_pcd(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud from PCD file."""
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {filepath}")
    return pcd


def load_reference_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load reference cloud from PCD, PLY, or OBJ format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reference cloud not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pcd":
        return load_pcd(filepath)
    elif ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(filepath)
        pcd = mesh.sample_points_uniformly(number_of_points=int(1e6))
        if len(pcd.points) == 0:
            raise ValueError(f"Empty mesh/point cloud: {filepath}")
        return pcd
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {filepath}")
        return pcd
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .pcd, .obj, or .ply")


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
            alpha = 0.8
        else:
            size = scale
            alpha = 0.3

        # Create small box for each pose
        box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        box.paint_uniform_color(color)
        box.transform(pose)

        # Connect poses with lines
        if i > 0:
            prev_pose = poses[::5][i - 1]
            line_points = np.array([prev_pose[:3, 3], pose[:3, 3]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color(color)
            geometries.append(line_set)

        geometries.append(box)

    return geometries


def _save_visualization_as_gif(
    geometries: list, output_path: str, fps: int = 10, duration_per_rotation: int = 1
):
    """Save visualization as GIF with rotating camera."""
    print(f"\nSaving visualization as GIF: {output_path}")

    # Extract trajectory center
    trajectory_points = []
    for geom in geometries:
        try:
            if hasattr(geom, "points"):
                points = np.asarray(geom.points)
                if len(points) > 0 and len(points) < 1000:
                    trajectory_points.extend(points)
            elif hasattr(geom, "vertices"):
                vertices = np.asarray(geom.vertices)
                if len(vertices) > 0 and len(vertices) < 1000:
                    trajectory_points.extend(vertices)
        except:
            pass

    if trajectory_points:
        trajectory_points = np.array(trajectory_points)
        traj_center = np.mean(trajectory_points, axis=0)
        print(f"  Trajectory center: {traj_center}")
    else:
        traj_center = np.array([0.0, 0.0, 0.0])
        print(f"  Warning: No trajectory points found, using origin")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, visible=False)

    for geom in geometries:
        vis.add_geometry(geom)

    # Setup camera view centered at trajectory
    ctr = vis.get_view_control()
    ctr.set_zoom(0.08)
    ctr.set_front([1, -0.0, -0.0])
    ctr.set_lookat(traj_center)
    ctr.set_up([-0.0, -0.0, 1.0])

    vis.poll_events()
    vis.update_renderer()

    # Capture frames with rotating camera
    num_frames = fps * duration_per_rotation
    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        for frame_idx in range(num_frames):
            # Rotate camera around vertical axis
            angle_increment = (360.0 / num_frames) * 5.8  # degrees to rotate() scale
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


def visualize_trajectories(
    traj_files: list,
    reference_cloud: str = None,
    save_gif: str = None,
    gif_fps: int = 10,
    gif_duration: int = 1,
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
    if reference_cloud and os.path.exists(reference_cloud):
        print(f"Loading reference cloud...")
        ref_cloud = load_reference_cloud(reference_cloud)
        geometries.append(ref_cloud)

    # Save as GIF or display interactively
    if save_gif:
        _save_visualization_as_gif(
            geometries, save_gif, fps=gif_fps, duration_per_rotation=gif_duration
        )
    else:
        print("\nVisualization legend:")
        for i in range(len(traj_files)):
            color_name = ["Green", "Red", "Blue", "Yellow", "Magenta", "Cyan"][i % 6]
            print(f"  {color_name}: Trajectory {i+1}")
        if reference_cloud:
            print(f"  Gray: Reference point cloud")
        print("\nDisplaying... (close window to exit)")

        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize trajectories with reference cloud"
    )
    parser.add_argument(
        "--traj", nargs="+", required=True, help="Trajectory file(s) to visualize"
    )
    parser.add_argument(
        "--reference", default=None, help="Reference point cloud (.pcd, .ply, .obj)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show interactive visualization"
    )
    parser.add_argument("--save-gif", default=None, help="Save as GIF to this path")
    parser.add_argument(
        "--gif-fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=7,
        help="Rotation duration in seconds (default: 1)",
    )

    args = parser.parse_args()

    # Default: visualize if no flags
    if not args.visualize and not args.save_gif:
        args.visualize = True

    visualize_trajectories(
        args.traj,
        args.reference,
        save_gif=args.save_gif,
        gif_fps=args.gif_fps,
        gif_duration=args.gif_duration,
    )
