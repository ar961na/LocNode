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
    is_basin_switch,
    load_transform,
    save_trajectory,
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

        # Config path: CONFIG_PATH env > ROS ~config param > default.
        config_path = os.environ.get(
            "CONFIG_PATH",
            rospy.get_param(
                "~config", "/opt/fastlio_localization/config/pipeline_config.yaml"
            ),
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
        self.rebase_rotation_deg = rospy.get_param(
            "~rebase_rotation_deg", config["alignment"].get("rebase_rotation_deg", 20.0)
        )
        self.rebase_translation_m = rospy.get_param(
            "~rebase_translation_m",
            config["alignment"].get("rebase_translation_m", 0.5),
        )
        # Optional initial-pose hint: enters alignment only as another
        # fitness-scored seed, so a wrong/absent value can't hurt results.
        try:
            self.initial_pose = self._parse_initial_pose(
                rospy.get_param(
                    "~initial_pose", config["alignment"].get("initial_pose")
                )
            )
        except Exception as e:
            rospy.logerr(f"Invalid initial_pose: {e}")
            sys.exit(1)
        self.cloud_topic = rospy.get_param(
            "~cloud_topic", config["topics"]["cloud_registered"]
        )
        self.odom_topic = rospy.get_param("~odom_topic", config["topics"]["odometry"])

        # Validate timing parameters: align_interval and map_accumulation_time
        # must be positive; warn when they exceed the run duration.
        self.duration = config["processing"].get("duration")
        for name, val in (
            ("align_interval", self.align_interval),
            ("map_accumulation_time", self.map_accumulation_time),
        ):
            if val is None or val <= 0:
                rospy.logerr(f"{name} must be > 0, got {val}")
                sys.exit(1)
        if self.duration is not None:
            if self.align_interval > self.duration:
                rospy.logwarn(
                    "align_interval (%ss) > duration (%ss): periodic alignment "
                    "may not tick; the final alignment still runs",
                    self.align_interval,
                    self.duration,
                )
            if self.map_accumulation_time > self.duration:
                rospy.logwarn(
                    "map_accumulation_time (%ss) > duration (%ss): the window "
                    "covers the whole run",
                    self.map_accumulation_time,
                    self.duration,
                )

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
        self.current_transform = None  # None until first successful alignment
        self.buffer_limit = 10000
        self.trajectory_buffer = deque(maxlen=self.buffer_limit)
        # Reference-frame trajectory: each pose transformed by the T_final
        # active when it arrived.
        self.trajectory_ref_buffer = deque(maxlen=self.buffer_limit)
        # Same per-cycle poses, but never rebuilt on a basin switch -> saved
        # as trajectory_reference_raw.txt (keeps the seams).
        self.trajectory_ref_raw_buffer = deque(maxlen=self.buffer_limit)
        # Guards the buffers + transform handover across the subscriber
        # callback threads and the Timer thread.
        self.buffer_lock = threading.Lock()
        self.reference_cloud = None
        self.reference_loaded = False
        # Set once finalize() runs, so shutdown can't align/save twice.
        self._finalized = False

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

    @staticmethod
    def _parse_initial_pose(value):
        """
        Parse the optional initial pose into a 4x4 map->reference matrix.

        Accepts None, a 4x4-matrix file path, a 7-element
        [x, y, z, qx, qy, qz, qw] start pose, or a 16-element matrix.
        """
        if value is None or value == "":
            return None
        if isinstance(value, str):
            T = load_transform(value)
        else:
            arr = np.array(value, dtype=float)
            if arr.shape == (7,):
                # Built from a (normalized) quaternion: rigid by construction.
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(arr[3:]).as_matrix()
                T[:3, 3] = arr[:3]
                return T
            if arr.shape not in ((16,), (4, 4)):
                raise ValueError(
                    f"initial_pose must be a file path, 7-element pose, or "
                    f"4x4 matrix; got shape {arr.shape}"
                )
            T = arr.reshape(4, 4)
        # Reject non-rigid matrices loudly rather than carry a dead seed.
        R = T[:3, :3]
        if (
            not np.allclose(R @ R.T, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4)
            or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        ):
            raise ValueError(
                "initial_pose is not a rigid transform (rotation block must be "
                "orthonormal with det +1 and bottom row [0, 0, 0, 1])"
            )
        return T

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
            points = np.array(
                list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))),
                dtype=np.float32,
            )
            if len(points) == 0:
                return
            with self.buffer_lock:
                self.cloud_buffer.append((now, points))
        except Exception as e:
            rospy.logerr(f"cloud_callback error: {e}")

    def odom_callback(self, msg):
        """Callback for odometry messages. Update trajectory buffers."""
        self.latest_odom = msg
        stamp = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        pose = [stamp, x, y, z, qx, qy, qz, qw]
        # Build the odometry matrix outside the lock, reused below for the
        # buffer row and the published message.
        T_odom = np.eye(4)
        T_odom[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T_odom[:3, 3] = [x, y, z]
        # One lock with timer_callback's handover so each pose lands in the
        # reference trajectory exactly once at first alignment.
        with self.buffer_lock:
            self.trajectory_buffer.append(pose)
            T = self.current_transform
            if T is not None:
                T_ref = T @ T_odom
                quat_ref = Rotation.from_matrix(T_ref[:3, :3]).as_quat()
                pose_ref = [stamp, *T_ref[:3, 3], *quat_ref]
                self.trajectory_ref_buffer.append(pose_ref)
                self.trajectory_ref_raw_buffer.append(pose_ref)

        # Publish transformed odometry if alignment available
        if self.reference_loaded and T is not None:
            odom_ref = self.make_odom_ref(msg, T_ref, quat_ref)
            self.odom_ref_pub.publish(odom_ref)
            self.broadcast_tf(odom_ref)

    def get_accumulated_cloud(self, now=None):
        """Accumulated cloud within the recent window.

        now defaults to sim time; finalize() passes the last cloud stamp
        (rospy time is unreliable at shutdown).
        """
        if now is None:
            now = rospy.get_time()
        cutoff = now - self.map_accumulation_time

        # Prune the window and snapshot under the lock so a concurrent append
        # from cloud_callback cannot mutate the deque mid-iteration.
        with self.buffer_lock:
            while self.cloud_buffer and self.cloud_buffer[0][0] < cutoff:
                self.cloud_buffer.popleft()
            if not self.cloud_buffer:
                return None
            all_points = [pts for _, pts in self.cloud_buffer]

        merged = np.vstack(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        return pcd

    def _snapshot(self, buffer):
        """Return a thread-safe shallow copy of a trajectory buffer."""
        with self.buffer_lock:
            return list(buffer)

    def timer_callback(self, event):
        """Periodic callback: align the recent window and publish live."""
        if not self.reference_loaded:
            rospy.logwarn("Reference not ready, alignment skipped")
            return

        now = rospy.get_time()
        accumulated = self.get_accumulated_cloud(now)
        if accumulated is None or len(accumulated.points) == 0:
            rospy.logwarn("No accumulated cloud, alignment skipped")
            return

        self._do_alignment(accumulated, publish=True, stamp=now)

    def _do_alignment(self, accumulated, publish, stamp):
        """Align accumulated to the reference, record the transform, and save.

        publish gates live ROS publishes; stamp is the handover/log time used
        at shutdown (live sim time otherwise).
        """
        rospy.loginfo(
            "Starting alignment: %d accumulated points", len(accumulated.points)
        )
        try:
            # Global alignment (coarse)
            T_global = self.global_align.align(accumulated, self.reference_pcd)
            fitness_global = self.global_align.get_fitness()
            rospy.loginfo(f"Global fitness: {fitness_global:.4f}")

            # Fine-tune. Seed ICP with the previous transform and the optional
            # initial pose (both quasi-static priors): align() scores each
            # against fresh RANSAC and warm-starts from the best, so a good
            # prior helps only by winning on fitness, never by fiat.
            candidate_seeds = [
                T for T in (self.current_transform, self.initial_pose) if T is not None
            ]
            # align() applies the keep/fallback policy internally and returns
            # the chosen transform; get_fitness() reflects that choice.
            T_final = self.local_align.align(
                accumulated,
                self.reference_pcd,
                T_global,
                candidate_seeds=candidate_seeds,
            )
            fitness_local = self.local_align.get_fitness()
            if self.local_align.has_diverged():
                rospy.logwarn("Local diverged, using seed transform")
            rospy.loginfo(
                f"Local fitness: {fitness_local:.4f} "
                f"(seed: {self.local_align.get_seed_fitness():.4f})"
            )

            # Handover under one lock. First alignment: backfill both buffers.
            # Basin switch: rebuild only the rebased buffer from raw odometry;
            # the raw buffer keeps its per-cycle seams.
            rebase_delta = None
            with self.buffer_lock:
                if self.current_transform is None and self.trajectory_buffer:
                    backfill = transform_trajectory(
                        list(self.trajectory_buffer), T_final
                    )
                    self.trajectory_ref_buffer.extend(backfill)
                    self.trajectory_ref_raw_buffer.extend(backfill)
                elif self.current_transform is not None:
                    rebase_delta = is_basin_switch(
                        self.current_transform,
                        T_final,
                        self.rebase_rotation_deg,
                        self.rebase_translation_m,
                    )
                    if rebase_delta is not None:
                        # Rare, bounded stall; the lock keeps the rebuild
                        # atomic with odom appends.
                        self.trajectory_ref_buffer.clear()
                        self.trajectory_ref_buffer.extend(
                            transform_trajectory(list(self.trajectory_buffer), T_final)
                        )
                self.current_transform = T_final
                # Handover stamp: live sim
                # time normally, the passed stamp at shutdown.
                align_stamp = (
                    stamp if rospy.is_shutdown() else rospy.Time.now().to_sec()
                )

            # Log before publishing so a publish failure can't lose the
            # record the offline rebuild needs.
            transform_flat = T_final.flatten().tolist()
            transform_str = " ".join([f"{x:.6f}" for x in transform_flat])
            log_entry = f"{align_stamp}, {len(accumulated.points)}, {fitness_global:.6f}, {fitness_local:.6f}, {transform_str}\n"
            with open(self.log_file, "a") as f:
                f.write(log_entry)

            if rebase_delta is not None:
                rospy.logwarn(
                    "Transform jumped %.1f deg / %.2f m: basin switch, "
                    "reference trajectory rebased onto new transform",
                    *rebase_delta,
                )
            save_transform(
                T_final, os.path.join(self.output_dir, "T_map_to_reference.txt")
            )

            # Live publishes only (skipped during shutdown finalize).
            if publish and not rospy.is_shutdown():
                self.transform_pub.publish(String(f"T_final:\n{T_final}"))
                self.fitness_pub.publish(Float32(fitness_local))

            # Save both: trajectory_reference.txt (rebased) and
            # trajectory_reference_raw.txt (per-cycle, with seams).
            traj_ref = self._snapshot(self.trajectory_ref_buffer)
            if traj_ref:
                save_trajectory(
                    traj_ref,
                    os.path.join(self.output_dir, "trajectory_reference.txt"),
                )
                rospy.loginfo(f"Trajectory saved: {len(traj_ref)} poses")
            traj_raw = self._snapshot(self.trajectory_ref_raw_buffer)
            if traj_raw:
                save_trajectory(
                    traj_raw,
                    os.path.join(self.output_dir, "trajectory_reference_raw.txt"),
                )

            rospy.loginfo("Alignment completed")

        except Exception as e:
            rospy.logerr(f"Alignment error: {e}")
            err_stamp = stamp if rospy.is_shutdown() else rospy.Time.now().to_sec()
            with open(self.log_file, "a") as f:
                f.write(f"{err_stamp}, ERROR, {str(e)}\n")

    def finalize(self):
        """Align the leftover tail and save, at shutdown.

        Guarantees a result even if the periodic timer never fired
        (align_interval >= bag duration). Runs once; no live publishes.
        """
        if self._finalized:
            return
        self._finalized = True
        try:
            if self.timer is not None:
                self.timer.shutdown()
            if not self.reference_loaded:
                rospy.logwarn("Reference never loaded; nothing to finalize")
                return
            with self.buffer_lock:
                last_stamp = self.cloud_buffer[-1][0] if self.cloud_buffer else None
            if last_stamp is None:
                rospy.logwarn("No clouds buffered; nothing to finalize")
                return
            accumulated = self.get_accumulated_cloud(last_stamp)
            if accumulated is None or len(accumulated.points) == 0:
                rospy.logwarn("No accumulated cloud for final alignment")
                return
            rospy.loginfo("Final alignment of leftover tail")
            self._do_alignment(accumulated, publish=False, stamp=last_stamp)
        except Exception as e:
            rospy.logwarn(f"Finalize failed: {e}")

    def make_odom_ref(self, odom_msg, T_ref, quat_ref):
        """Build a reference-frame Odometry message from a precomputed pose."""
        odom_ref = Odometry()
        odom_ref.header = odom_msg.header
        odom_ref.header.frame_id = "reference"
        odom_ref.child_frame_id = odom_msg.child_frame_id
        odom_ref.pose.pose.position.x = T_ref[0, 3]
        odom_ref.pose.pose.position.y = T_ref[1, 3]
        odom_ref.pose.pose.position.z = T_ref[2, 3]
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


def main():
    """Main entry point: init node, spin, then finalize on shutdown."""
    try:
        node = LocalizationNode()
        rospy.spin()
        # SIGINT after playback makes spin() return; align the tail and save.
        node.finalize()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
