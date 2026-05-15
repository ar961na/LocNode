# Stage 1: Get uv binary for fast Python package installation
FROM ghcr.io/astral-sh/uv:latest AS uv-bin

# Stage 2: FAST-LIO2 + localization image
FROM kenny0407/marslab_fastlio2:latest

WORKDIR /opt/fastlio_localization

# Copy uv package manager from official image
COPY --from=uv-bin /uv /uvx /bin/

# Switch to root for installation
USER root

ENV PYTHONPATH="/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages"

# Install Python dependencies
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN uv pip install --system --break-system-packages -r /tmp/requirements-docker.txt && \
    python3 -c "import open3d, numpy, scipy, cv2; print('✓ Dependencies OK')"

# Fix ROS GPG key expiration issue
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F42ED6FBAB17C654

# Install livox_ros_driver2 for Livox LiDAR support
RUN apt-get update && apt-get install -y git build-essential \
    && mkdir -p /home/mars_ugv/livox_ws/src \
    && cd /home/mars_ugv/livox_ws/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
    && cd .. \
    && /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make" \
    && echo "source /home/mars_ugv/livox_ws/devel/setup.bash" >> /home/mars_ugv/.bashrc

# Copy application code
COPY nodes/ /opt/fastlio_localization/nodes/
#COPY docker/pipeline_all_in_docker.py /opt/fastlio_localization/
COPY docker/docker-entrypoint.sh /opt/fastlio_localization/bin/docker-entrypoint.sh
COPY launch/ /opt/fastlio_localization/launch/
COPY config/ /opt/fastlio_localization/config/
COPY scripts/downsample_reference.py /opt/fastlio_localization/scripts/downsample_reference.py

RUN chmod +x /opt/fastlio_localization/scripts/downsample_reference.py

# Create directories for volumes
RUN mkdir -p /data /results /opt/fastlio_localization/bin

# Set permissions and switch to non-root user
RUN chown -R mars_ugv:mars_ugv /opt/fastlio_localization /data /results && \
    chmod +x /opt/fastlio_localization/bin/docker-entrypoint.sh
USER mars_ugv

# Entry point for container
ENTRYPOINT ["/opt/fastlio_localization/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]