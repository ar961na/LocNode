#!/usr/bin/env python3
"""
Unified visualization script for point clouds and trajectories.

Supports:
- Point cloud visualization (PCD, PLY, OBJ formats)
- Trajectory visualization (TUM format with multiple trajectories)
- Clouds and trajectories overlaid together
- GIF export with rotating camera view
- Interactive display

Usage examples:
    # Point cloud as rotating GIF
    python visualize.py --cloud cloud.ply --output animation.gif --fps 15 --duration 5

    # Trajectory overlaid on a map cloud
    python visualize.py --traj trajectory.txt --cloud map.ply --output traj.gif

    # Multiple trajectories on a map, interactive
    python visualize.py --traj traj1.txt traj2.txt --cloud map.ply --visualize

    # Just a point cloud, interactive
    python visualize.py --cloud cloud.ply --visualize
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import colorsys
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from PIL import Image

# Color applied to clouds shown as background behind trajectories.
CLOUD_BG_COLOR = [0.7, 0.7, 0.7]


# --- Point Cloud Loading ---
def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from PCD, PLY, or OBJ format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(filepath)
    elif ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(filepath)
        pcd = mesh.sample_points_uniformly(number_of_points=int(1e6))
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .pcd, .ply, or .obj")

    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {filepath}")

    return pcd


# --- Trajectory Loading and Conversion ---
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
    trajectory: list, color: list, scale: float = 0.02, traj_downsample: int = 5
) -> list:
    """Create 3D visualization geometries (boxes + lines) from trajectory."""
    geometries = []
    poses = trajectory_to_poses(trajectory)

    # Downsample trajeectory.
    sampled_poses = poses[::traj_downsample]
    for i, pose in enumerate(sampled_poses):
        size = scale

        # Create small box for each pose
        box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        box.paint_uniform_color(color)
        box.transform(pose)
        geometries.append(box)

        # Connect poses with lines
        if i > 0:
            prev_pose = sampled_poses[i - 1]
            line_points = np.array([prev_pose[:3, 3], pose[:3, 3]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color(color)
            geometries.append(line_set)

    return geometries


def generate_colors(n: int) -> list:
    """Generate n visually distinct RGB colors (0-1 range) by evenly spacing hue."""
    return [list(colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 0.9)) for i in range(n)]


# --- GIF Generation ---
def compute_scene_bounds(geometries: list):
    """Combined center and max axis extent across trajectories, via bounding boxes."""
    min_bounds, max_bounds = [], []
    for geom in geometries:
        try:
            bbox = geom.get_axis_aligned_bounding_box()
            min_bounds.append(bbox.get_min_bound())
            max_bounds.append(bbox.get_max_bound())
        except Exception:
            pass

    if not min_bounds:
        return np.zeros(3), 1.0

    min_bound = np.min(min_bounds, axis=0)
    max_bound = np.max(max_bounds, axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = float(np.max(max_bound - min_bound))
    return center, extent


def save_rotating_gif(
    geometries: list,
    output_path: str,
    fps: int = 10,
    duration: int = 5,
    width: int = 1024,
    height: int = 768,
    center: np.ndarray = None,
    extent: float = None,
    focus_geometries: list = None,
    zoom: float = 0.1,
):
    """
    Save a rotating GIF of geometries using rotating camera.
    Uses Z-up coordinate system.

    Args:
        geometries: List of Open3D geometries to render
        output_path: Output GIF file path
        fps: Frames per second
        duration: Duration of full rotation in seconds
        width: Output image width
        height: Output image height
        center: Optional center point for camera lookat
        extent: Optional extent for zoom calculation
        focus_geometries: Optional subset of geometries to frame around
    """
    print(f"\nGenerating rotating GIF: {output_path}")

    # Compute center and extent from focus_geometries if not provided.
    # When focus_geometries is set (e.g., trajectory-only), frame around those
    # while rendering all geometries.
    if center is None or extent is None:
        bounds_geoms = focus_geometries if focus_geometries else geometries
        auto_center, auto_extent = compute_scene_bounds(bounds_geoms)
        if center is None:
            center = auto_center
        if extent is None:
            extent = auto_extent

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
    ctr.set_zoom(zoom / extent if extent > 0 else zoom)
    ctr.set_front([1, 0, 0])  # Looking along +X initially
    ctr.set_up([0, 0, 1])  # Z is up

    vis.poll_events()
    vis.update_renderer()

    # Capture frames with rotating camera
    num_frames = fps * duration
    frames = []

    try:
        for frame_idx in range(num_frames):
            # Rotate around the vertical (Z) axis
            angle_increment = (360.0 / num_frames) * 5.8
            ctr.rotate(angle_increment, 0)

            vis.poll_events()
            vis.update_renderer()

            try:
                buf = vis.capture_screen_float_buffer(do_render=True)
                frames.append(Image.fromarray((np.asarray(buf) * 255).astype(np.uint8)))
            except Exception as e:
                print(f"  Warning: Failed to capture frame {frame_idx}: {e}")

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


# --- Main Visualization Logic ---
def visualize_scene(
    cloud_files: list = None,
    traj_files: list = None,
    output_gif: str = None,
    fps: int = 10,
    duration: int = 5,
    width: int = 1024,
    height: int = 768,
    visualize: bool = False,
    cloud_bg: bool = False,
    traj_downsample: int = 5,
    zoom: float = 0.1,
):
    """Visualize any combination of point clouds and trajectories together."""
    cloud_files = cloud_files or []
    traj_files = traj_files or []

    geometries = []
    trajectory_geometries = []

    # Trajectories get vivid, distinct colors.
    traj_colors = generate_colors(len(traj_files))
    for traj_file, color in zip(traj_files, traj_colors):
        traj = load_trajectory(traj_file)
        print(f"  Trajectory: {len(traj)} poses from {os.path.basename(traj_file)}")
        if traj:
            traj_geoms = create_trajectory_visualization(
                traj, color, traj_downsample=traj_downsample
            )
            geometries.extend(traj_geoms)
            trajectory_geometries.extend(traj_geoms)

    # If cloud_bg is True, paint clouds in uniform gray to serve as background for trajectories.
    for cloud_file in cloud_files:
        print(f"Loading point cloud {os.path.basename(cloud_file)}...")
        pcd = load_point_cloud(cloud_file)
        print(f"  Loaded {len(pcd.points)} points")
        if cloud_bg:
            pcd.paint_uniform_color(CLOUD_BG_COLOR)
        geometries.append(pcd)

    if not geometries:
        print("Nothing to visualize")
        return

    # Save as GIF or display
    if output_gif:
        # When both clouds and trajectories exist, frame around trajectory only.
        focus_geoms = trajectory_geometries if trajectory_geometries else None
        save_rotating_gif(
            geometries,
            output_gif,
            fps=fps,
            duration=duration,
            width=width,
            height=height,
            focus_geometries=focus_geoms,
            zoom=zoom,
        )
    elif visualize:
        if traj_files:
            print("\nVisualization legend:")
            for traj_file, color in zip(traj_files, traj_colors):
                rgb = ", ".join(f"{c:.2f}" for c in color)
                print(f"  RGB({rgb}): {os.path.basename(traj_file)}")
        if cloud_files:
            label = "gray" if cloud_bg else "natural color"
            print(f"  Point cloud(s) shown in {label}")
        o3d.visualization.draw_geometries(geometries)
    else:
        print("Use --output to save as GIF or --visualize to display interactively")


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Unified visualization for point clouds and trajectories"
    )

    # Inputs (at least one required; both may be combined)
    parser.add_argument(
        "--cloud", "-c", nargs="+", help="Point cloud file(s) (.pcd, .ply, .obj)"
    )
    parser.add_argument(
        "--traj", "-t", nargs="+", help="Trajectory file(s) in TUM format"
    )

    # Output options
    output_group = parser.add_argument_group("Output (choose one)")
    output_group.add_argument("--output", "-o", help="Save as GIF to this path")
    output_group.add_argument(
        "--visualize", "-v", action="store_true", help="Show interactive visualization"
    )

    # Visualization options
    vis_group = parser.add_argument_group("Visualization options")
    vis_group.add_argument(
        "--cloud-bg", action="store_true", help="Show clouds as gray background"
    )
    vis_group.add_argument(
        "--traj-downsample",
        type=int,
        default=5,
        help="Downsample trajectory poses for visualization (default: 5)",
    )
    vis_group.add_argument(
        "--zoom",
        type=float,
        default=0.1,
        help="Zoom level for visualization (default: 0.1)",
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
        parser.error("Please provide --cloud and/or --traj")
    if args.output and args.visualize:
        parser.error("Please choose only one of --output or --visualize")
    if args.zoom <= 0:
        parser.error("Zoom level must be a positive number")
    if args.traj_downsample <= 0:
        parser.error("Trajectory downsample must be a positive integer")
    if args.fps <= 0:
        parser.error("FPS must be a positive integer")
    if args.duration <= 0:
        parser.error("Duration must be a positive integer")
    if args.width <= 0 or args.height <= 0:
        parser.error("Width and height must be positive integers")

    # Default to visualize if no output specified
    if not args.output and not args.visualize:
        args.visualize = True

    visualize_scene(
        cloud_files=args.cloud,
        traj_files=args.traj,
        output_gif=args.output,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
        visualize=args.visualize,
        cloud_bg=args.cloud_bg,
        traj_downsample=args.traj_downsample,
        zoom=args.zoom,
    )


if __name__ == "__main__":
    main()
