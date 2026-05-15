#!/bin/bash
# Docker entrypoint script for FAST-LIO + Localization pipeline
# Starts ROS components and runs localization node

set -e

# Source ROS setup
source /opt/ros/noetic/setup.bash
source /home/mars_ugv/catkin_ws/devel/setup.bash

# Configure environment for headless rendering
export PYTHONPATH="/opt/fastlio_localization:${PYTHONPATH}"
export OPEN3D_HEADLESS_RENDERING=1
export QT_QPA_PLATFORM=offscreen

# Start ROS master in background
roscore &
sleep 3

# Use simulated time for rosbag playback
rosparam set /use_sim_time true

# Start FAST-LIO mapping (no RViz visualization)
roslaunch fast_lio mapping_mid360.launch rviz:=false &
sleep 10

# Start localization node (blocks until shutdown)
python3 /opt/fastlio_localization/nodes/localization_node.py

# Wait for background processes
sleep 10
wait