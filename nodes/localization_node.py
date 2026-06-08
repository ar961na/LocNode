#!/usr/bin/env python3
"""
ROS-based localization node for FAST-LIO + reference cloud alignment.

Accumulates FAST-LIO point clouds and performs periodic global+local alignment
to reference cloud. Outputs transformed odometry in reference frame.
"""

import sys

sys.path.insert(0, "/opt/fastlio_localization")

import os

os.environ["OPEN3D_HEADLESS_RENDERING"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import rospy
import numpy as np
import open3d as o3d
import yaml
import threading
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, Float32
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from scipy.spatial.transform import Rotation
from collections import deque

from nodes.common import (
    GlobalAlignMethod,
    LocalAlignMethod,
    PointCloudProcessor,
    save_transform,
    transform_trajectory,
)
from nodes.global_align import GlobalAlign
from nodes.local_align import LocalAlign


class LocalizationNode:
    """ROS node for cloud-based localization with reference alignment."""

    def __init__(self):
        """Initialize ROS node, load configuration, and setup subscribers/publishers."""
        rospy.init_node("localization_node", anonymous=False)

        # Load configuration
        config_path = rospy.get_param(
            "~config", "/opt/fastlio_localization/config/pipeline_config.yaml"
        )
        if not os.path.exists(config_path):
            rospy.logerr(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load parameters (priority: environment variable > ROS param > config file)
        default_ref_pcd = os.path.join("/data", config["data"]["reference_cloud"])
        self.reference_pcd = os.environ.get(
            "REFERENCE_PCD", rospy.get_param("~reference_pcd", default_ref_pcd)
        )
        self.output_dir = os.environ.get(
            "OUTPUT_DIR", rospy.get_param("~output_dir", config["output"]["dir"])
        )
        self.align_interval = rospy.get_param(
            "~align_interval", config["processing"]["align_interval"]
        )
        self.map_accumulation_time = rospy.get_param(
            "~map_accumulation_time", config["processing"]["map_accumulation_time"]
        )
        self.voxel_size_global = rospy.get_param(
            "~voxel_size_global", config["alignment"]["voxel_size_global"]
        )
        self.voxel_size_local = rospy.get_param(
            "~voxel_size_local", config["alignment"]["voxel_size_local"]
        )
        self.icp_threshold = rospy.get_param(
            "~icp_threshold", config["alignment"]["icp_threshold"]
        )
        self.cloud_topic = rospy.get_param(
            "~cloud_topic", config["topics"]["cloud_registered"]
        )
        self.odom_topic = rospy.get_param("~odom_topic", config["topics"]["odometry"])

        # Validate reference cloud exists
        if not os.path.exists(self.reference_pcd):
            rospy.logerr(f"Reference cloud not found: {self.reference_pcd}")
            sys.exit(1)

        # Setup output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.maps_dir = os.path.join(self.output_dir, "cached_maps")
        os.makedirs(self.maps_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "alignment_log.txt")
        with open(self.log_file, "w") as f:
            f.write(
                "# timestamp, accumulated_points, global_fitness, local_fitness, transform(4x4 flattened)\n"
            )

        # Initialize aligners
        self.global_align = GlobalAlign(
            voxel_size=self.voxel_size_global, method=GlobalAlignMethod.FPFH_RANSAC
        )
        self.local_align = LocalAlign(
            voxel_size=self.voxel_size_local,
            threshold=self.icp_threshold,
            method=LocalAlignMethod.P2PLANE_ICP,
        )

        # State buffers
        self.cloud_buffer = deque()
        self.latest_odom = None
        self.current_transform = np.eye(4)
        self.trajectory_buffer = []
        self.reference_cloud = None
        self.reference_loaded = False
        self.buffer_limit = 10000

        # Publishers
        self.transform_pub = rospy.Publisher(
            config["topics"]["transform"], String, queue_size=1
        )
        self.fitness_pub = rospy.Publisher(
            config["topics"]["fitness"], Float32, queue_size=1
        )
        self.odom_ref_pub = rospy.Publisher(
            config["topics"]["odometry_ref"], Odometry, queue_size=10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.status_pub = rospy.Publisher("/localization/status", String, queue_size=1)

        # Subscribers
        self.cloud_sub = rospy.Subscriber(
            self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=10
        )

        # Alignment timer (started after reference loads)
        self.timer = None

        # Load reference cloud in background
        self.load_thread = threading.Thread(target=self._load_reference)
        self.load_thread.start()

        # Check reference status and start alignment timer
        rospy.Timer(rospy.Duration(1.0), self._check_reference_and_start, oneshot=True)

        # Publish dummy odometry to register topic in RViz
        test_odom = Odometry()
        test_odom.header.frame_id = "reference"
        test_odom.child_frame_id = "base_link"
        test_odom.header.stamp = rospy.Time.now()
        self.odom_ref_pub.publish(test_odom)

        rospy.loginfo("LocalizationNode initialized, loading reference cloud...")

    def _load_reference(self):
        """Load reference cloud in background thread."""
        try:
            self.reference_cloud = PointCloudProcessor.load_reference_cloud(
                self.reference_pcd
            )
            self.reference_loaded = True
            self.status_pub.publish("READY")
            rospy.loginfo(
                f"Reference cloud loaded: {len(self.reference_cloud.points)} points"
            )
            with open(self.log_file, "a") as f:
                f.write(
                    f"{rospy.Time.now().to_sec()}, reference_loaded, points={len(self.reference_cloud.points)}\n"
                )
        except Exception as e:
            rospy.logerr(f"Failed to load reference cloud: {e}")
            self.status_pub.publish("ERROR")

    def _check_reference_and_start(self, event):
        """Check if reference loaded; start alignment timer when ready."""
        if self.reference_loaded:
            if self.timer is None:
                self.timer = rospy.Timer(
                    rospy.Duration(self.align_interval), self.timer_callback
                )
                rospy.loginfo("Reference ready, alignment timer started")
        else:
            # Retry in 1 second
            rospy.Timer(
                rospy.Duration(1.0), self._check_reference_and_start, oneshot=True
            )

    def cloud_callback(self, msg):
        """Callback for point cloud messages. Buffer points for accumulation."""
        try:
            now = msg.header.stamp.to_sec()
            points = []
            for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
                points.append([p[0], p[1], p[2]])
            if not points:
                return
            self.cloud_buffer.append((now, np.array(points)))
        except Exception as e:
            rospy.logerr(f"cloud_callback error: {e}")

    def odom_callback(self, msg):
        """Callback for odometry messages. Update trajectory buffer."""
        self.latest_odom = msg
        stamp = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.trajectory_buffer.append([stamp, x, y, z, qx, qy, qz, qw])

        # Limit buffer size
        if len(self.trajectory_buffer) > self.buffer_limit:
            self.trajectory_buffer.pop(0)

        # Publish transformed odometry if alignment available
        if self.reference_loaded and not np.array_equal(
            self.current_transform, np.eye(4)
        ):
            odom_ref = self.transform_odometry(msg, self.current_transform)
            self.odom_ref_pub.publish(odom_ref)
            self.broadcast_tf(odom_ref)

    def get_accumulated_cloud(self):
        """Get accumulated point cloud within recent time window."""
        now = rospy.get_time()
        cutoff = now - self.map_accumulation_time

        # Remove old points outside window
        while self.cloud_buffer and self.cloud_buffer[0][0] < cutoff:
            self.cloud_buffer.popleft()

        if not self.cloud_buffer:
            return None

        # Merge all buffered points
        all_points = []
        for _, pts in self.cloud_buffer:
            all_points.append(pts)
        merged = np.vstack(all_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        return pcd

    def timer_callback(self, event):
        """Periodic callback for global + local alignment."""
        if not self.reference_loaded:
            rospy.logwarn("Reference not ready, alignment skipped")
            return

        accumulated = self.get_accumulated_cloud()
        if accumulated is None or len(accumulated.points) == 0:
            rospy.logwarn("No accumulated cloud, alignment skipped")
            return

        rospy.loginfo(
            "Starting alignment: %d accumulated points", len(accumulated.points)
        )
        try:
            # Save accumulated cloud temporarily
            temp_map = "/tmp/accumulated_map.pcd"
            o3d.io.write_point_cloud(temp_map, accumulated)

            # Global alignment (coarse)
            T_global = self.global_align.align(temp_map, self.reference_pcd)
            fitness_global = self.global_align.get_fitness()
            rospy.loginfo(f"Global fitness: {fitness_global:.4f}")

            # Local alignment (fine-tune from global)
            T_final = self.local_align.align(temp_map, self.reference_pcd, T_global)
            fitness_local = self.local_align.get_fitness()
            rospy.loginfo(f"Local fitness: {fitness_local:.4f}")

            # Use global if local worse (divergence)
            if fitness_local < fitness_global:
                rospy.logwarn("Local diverged, using global transform")
                T_final = T_global
                fitness_local = fitness_global

            self.current_transform = T_final
            save_transform(
                T_final, os.path.join(self.output_dir, "T_map_to_reference.txt")
            )

            # Publish results
            self.transform_pub.publish(String(f"T_final:\n{T_final}"))
            self.fitness_pub.publish(Float32(fitness_local))

            # Log alignment
            transform_flat = T_final.flatten().tolist()
            transform_str = " ".join([f"{x:.6f}" for x in transform_flat])
            log_entry = f"{rospy.Time.now().to_sec()}, {len(accumulated.points)}, {fitness_global:.6f}, {fitness_local:.6f}, {transform_str}\n"
            with open(self.log_file, "a") as f:
                f.write(log_entry)

            # Save trajectory in reference frame
            if self.trajectory_buffer:
                traj_ref = transform_trajectory(self.trajectory_buffer, T_final)
                out_file = os.path.join(self.output_dir, "trajectory_reference.txt")
                self.save_trajectory(traj_ref, out_file)
                rospy.loginfo(f"Trajectory saved: {len(traj_ref)} poses")

            rospy.loginfo("Alignment completed")

        except Exception as e:
            rospy.logerr(f"Alignment error: {e}")
            with open(self.log_file, "a") as f:
                f.write(f"{rospy.Time.now().to_sec()}, ERROR, {str(e)}\n")

    def transform_odometry(self, odom_msg, T):
        """Transform odometry pose to reference frame using 4x4 matrix."""
        pos = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ]
        )
        quat = np.array(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )

        T_odom = np.eye(4)
        T_odom[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T_odom[:3, 3] = pos
        T_ref = T @ T_odom

        # Create transformed odometry message
        odom_ref = Odometry()
        odom_ref.header = odom_msg.header
        odom_ref.header.frame_id = "reference"
        odom_ref.child_frame_id = odom_msg.child_frame_id
        odom_ref.pose.pose.position.x = T_ref[0, 3]
        odom_ref.pose.pose.position.y = T_ref[1, 3]
        odom_ref.pose.pose.position.z = T_ref[2, 3]
        quat_ref = Rotation.from_matrix(T_ref[:3, :3]).as_quat()
        odom_ref.pose.pose.orientation.x = quat_ref[0]
        odom_ref.pose.pose.orientation.y = quat_ref[1]
        odom_ref.pose.pose.orientation.z = quat_ref[2]
        odom_ref.pose.pose.orientation.w = quat_ref[3]
        odom_ref.pose.covariance = odom_msg.pose.covariance

        return odom_ref

    def broadcast_tf(self, odom_ref):
        """Broadcast transformation to TF tree."""
        t = TransformStamped()
        t.header = odom_ref.header
        t.child_frame_id = odom_ref.child_frame_id
        t.transform.translation.x = odom_ref.pose.pose.position.x
        t.transform.translation.y = odom_ref.pose.pose.position.y
        t.transform.translation.z = odom_ref.pose.pose.position.z
        t.transform.rotation = odom_ref.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def save_trajectory(trajectory, filepath):
        """Save trajectory to TUM format file."""
        with open(filepath, "w") as f:
            for stamp, x, y, z, qx, qy, qz, qw in trajectory:
                f.write(
                    f"{stamp:.9f} {x:.8f} {y:.8f} {z:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
                )


def main():
    """Main entry point. Initialize node and spin."""
    try:
        node = LocalizationNode()
        rospy.spin()

        # Save final results on shutdown
        if node.current_transform is not None:
            save_transform(
                node.current_transform,
                os.path.join(node.output_dir, "T_map_to_reference.txt"),
            )
            rospy.loginfo("Final transform saved")
        if node.trajectory_buffer:
            traj_ref = transform_trajectory(
                node.trajectory_buffer, node.current_transform
            )
            out_file = os.path.join(node.output_dir, "trajectory_reference.txt")
            node.save_trajectory(traj_ref, out_file)
            rospy.loginfo("Final trajectory saved")

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
