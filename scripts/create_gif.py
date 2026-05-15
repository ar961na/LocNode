#!/usr/bin/env python3
"""
Create a rotating GIF from a single point cloud (PCD, PLY, OBJ).

Usage:
    python pcd_to_gif.py --input cloud.ply --output animation.gif --fps 15 --duration 5
"""

import argparse
import os
import tempfile
import shutil

import numpy as np
import open3d as o3d
from PIL import Image


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from various formats."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pcd":
        pcd = o3d.io.read_point_cloud(filepath)
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(filepath)
    elif ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(filepath)
        pcd = mesh.sample_points_uniformly(number_of_points=1000000)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .pcd, .ply, or .obj")

    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {filepath}")

    return pcd


def save_rotating_gif(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
    fps: int = 15,
    duration: int = 5,
    width: int = 1024,
    height: int = 768,
):
    """
    Save a rotating GIF of the point cloud using the rotate() method.
    Uses Z‑up coordinate system to match the trajectory visualizer.
    """
    print(f"\nGenerating rotating GIF: {output_path}")

    # ------------------------------------------------------------------
    # 1. Compute center and camera distance from bounding box
    # ------------------------------------------------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_max_extent()

    print(f"  Cloud center: {center}")
    print(f"  Cloud extent: {extent:.3f}")

    # ------------------------------------------------------------------
    # 2. Create visualizer and set initial view (Z‑up)
    # ------------------------------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_zoom(0.8 / extent if extent > 0 else 0.8)
    # Use Z as up (vertical) and look from a slight angle
    ctr.set_front([1, 0, 0])  # looking along +X initially
    ctr.set_up([0, 0, 1])  # Z is up

    vis.poll_events()
    vis.update_renderer()

    # ------------------------------------------------------------------
    # 3. Capture frames while rotating around the vertical (Z) axis
    # ------------------------------------------------------------------
    num_frames = fps * duration
    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        for frame_idx in range(num_frames):
            # Rotate around the camera's up axis (now Z)
            angle_increment = (360.0 / num_frames) * 5.8
            ctr.rotate(angle_increment, 0)

            vis.poll_events()
            vis.update_renderer()

            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            vis.capture_screen_image(frame_path, do_render=True)

            try:
                frame_image = Image.open(frame_path)
                frames.append(frame_image)
            except Exception as e:
                print(f"  Warning: Failed to load frame {frame_idx}: {e}")

            if (frame_idx + 1) % 10 == 0:
                print(f"  Captured {frame_idx + 1}/{num_frames} frames")

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


def main():
    parser = argparse.ArgumentParser(
        description="Create a rotating GIF from a point cloud."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input point cloud file (.pcd, .ply, .obj)"
    )
    parser.add_argument("--output", "-o", required=True, help="Output GIF file path")
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration of full rotation in seconds (default: 5)",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Output image width (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=768, help="Output image height (default: 768)"
    )

    args = parser.parse_args()

    pcd = load_point_cloud(args.input)
    save_rotating_gif(
        pcd,
        args.output,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
