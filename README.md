# FAST-LIO2 + Localization Pipeline

A Docker-based LiDAR localization pipeline combining FAST-LIO2 mapping with point cloud alignment to a reference model.

## Prerequisites

- **PyYAML**

- **Docker**
- **Rosbag** data file with Livox MID-360 topics (.bag format)
- **Reference point cloud** (PLY/PCD format for alignment target)

## Project Structure

```
LocNode/
├── config/
│   └── pipeline_config.yaml          # Pipeline configuration (alignment params, topics, paths)
├── data/
│   └── location/                    # Data directory (create this structure)
│       ├── rosbag.bag              # ROS bag file (input)
│       └── reference.ply           # Reference cloud for alignment (input)
├── docker/
│   ├── Dockerfile                    # Container build configuration
│   ├── docker-entrypoint.sh         # Container startup script
│   ├── docker_build_and_run.sh       # Main orchestration script (RUN THIS)
│   └── requirements-docker.txt       # Python dependencies
├── launch/
│   └── localization.launch           # ROS launch file
├── nodes/
│   ├── __init__.py                   # Module exports
│   ├── common.py                     # Utilities: enums, PointCloudProcessor, caching
│   ├── global_align.py              # FPFH-RANSAC global alignment
│   ├── local_align.py               # ICP local alignment
│   └── localization_node.py         # Main ROS node
├── results/
│   └── run_YYYYMMDD_HHMMSS/         # Output (created per run)
│       ├── alignment_log.txt         # Detailed alignment metrics
│       ├── trajectory_reference.txt  # Final trajectory (TUM format)
│       ├── T_map_to_reference.txt    # Transformation matrix (4x4)
│       ├── pipeline_config.yaml      # Config snapshot
│       └── cached_maps/              # FPFH features cache
└── scripts/
    ├── downsample_reference.py       # Utility to downsample point clouds
    └── visualize.py                  # Unified visualization (point clouds & trajectories)
```

## Data Folder Setup

Create your data structure in `data/`:

```bash
mkdir -p data/location
cp /path/to/your/data.bag data/location/
cp /path/to/your/reference.ply data/location/
```

Supported formats:
- **Bag files**: ROS rosbag format with Livox MID-360 topics (.bag)
- **Point clouds**: PLY, PCD, OBJ mesh formats
- **Trajectories**: TUM format (timestamp + position + quaternion)

## Configuration

Edit `config/pipeline_config.yaml` to customize:

### Data Paths
```yaml
data:
  bag: "location/your_data.bag"           # Rosbag relative to /data
  reference_cloud: "location/ref.ply"     # Reference for alignment
  downsampled_reference_cloud: "location/ref_downsampled.ply"
```

### Processing Parameters
```yaml
processing:
  duration: 40                  # Extract this many seconds from bag
  align_interval: 2.0           # Run alignment every N seconds
  map_accumulation_time: 2.0    # Accumulate clouds from last N seconds
```

### Alignment Parameters
```yaml
alignment:
  voxel_size_global: 0.4        # Downsampling for FPFH (larger = faster)
  voxel_size_local: 0.1         # Downsampling for ICP
  icp_threshold: 0.5            # Correspondence distance threshold (meters)
  global_method: "FPFH_RANSAC"  # Coarse alignment
  local_method: "P2PLANE_ICP"   # Fine alignment (more robust than P2P_ICP)
```

### ROS Topics
```yaml
topics:
  cloud_registered: "/cloud_registered"  # Input: FAST-LIO point cloud
  odometry: "/Odometry"                   # Input: FAST-LIO odometry
  odometry_ref: "/localization/odometry_ref"     # Output: transformed odometry
  transform: "/localization/transform"            # Output: T_ref_to_map
  fitness: "/localization/fitness"                # Output: alignment fitness
```

## Quick Start

### 1. Prepare Your Data

```bash
# Create data directory
mkdir -p data/location

# Copy your files
cp /path/to/bagfile.bag data/location/
cp /path/to/reference.ply data/location/
```

### 2. Configure Pipeline

Edit `config/pipeline_config.yaml`:
```yaml
data:
  bag: "E-2004-A2/bagfile.bag"
  reference_cloud: "E-2004-A2/reference.ply"
```

### 3. Run Pipeline

```bash
# Build Docker image and execute pipeline
bash docker/docker_build_and_run.sh
```

The script will:
- Build Docker image with FAST-LIO2 and localization
- Verify reference cloud exists
- Downsample reference if not already done
- Play rosbag and run alignment
- Save results to `results/run_YYYYMMDD_HHMMSS/`

### 4. View Results

```bash
# Check trajectory
head results/run_YYYYMMDD_HHMMSS/trajectory_reference.txt

# View detailed metrics
cat results/run_YYYYMMDD_HHMMSS/alignment_log.txt

# Visualize trajectory over the reference cloud
python3 scripts/visualize.py \
  --traj results/run_YYYYMMDD_HHMMSS/trajectory_reference.txt \
  --cloud data/location/reference.ply --cloud-bg --visualize
```

## Output Files

Each run creates a timestamped directory with:

### `trajectory_reference.txt`
**TUM trajectory format** (8 columns):
```
timestamp x y z qx qy qz qw
1775050482.372277737 -1.13308480 -0.39594947 0.09314294 -0.15809042 0.06894059 0.88799696 0.42628162
```
- `timestamp`: ROS time in seconds
- `x y z`: position in reference frame (meters)
- `qx qy qz qw`: unit quaternion (x, y, z, w components)

### `T_map_to_reference.txt`
**4×4 transformation matrix** (flattened):
```
-0.651746 -0.752118 0.097703 -1.735784 0.755234 -0.655413 -0.007450 0.074499 ...
```
Converts from map frame to reference frame: `p_ref = T @ p_map`

### `alignment_log.txt`
**Detailed metrics** (CSV format):
```
timestamp, accumulated_points, global_fitness, local_fitness, transform(4x4 flattened)
0.0, reference_loaded, points=2432896
1775050490.2707038, 108041, 0.654332, 0.916432, -0.651746 -0.752118 ...
```
- `accumulated_points`: Number of points buffered for alignment
- `global_fitness`: RANSAC match quality (higher = better)
- `local_fitness`: ICP final fitness score
- Remaining columns: 16-element transformation matrix

### `pipeline_config.yaml`
Snapshot of configuration used for this run (for reproducibility)

### Downsample Point Clouds

```bash
python3 scripts/downsample_reference.py \
  --input reference.ply \
  --output reference_downsampled.ply \
  --voxel_size 0.1
```

### Visualize Trajectories and Point Clouds

**Unified visualization script** for trajectories and point clouds, separately or
together. Both `--cloud` and `--traj` accept multiple files and may be combined.

```bash
# Trajectory over the reference cloud (cloud muted to gray background)
python3 scripts/visualize.py \
  --traj trajectory.txt \
  --cloud reference.ply --cloud-bg \
  --visualize

# Save a trajectory + reference as a rotating GIF (framed on the trajectory)
python3 scripts/visualize.py \
  --traj trajectory.txt \
  --cloud reference.ply --cloud-bg \
  --output trajectory.gif --duration 3

# Compare multiple trajectories (each gets a distinct auto-generated color)
python3 scripts/visualize.py \
  --traj trajectory1.txt trajectory2.txt \
  --cloud reference.ply --cloud-bg \
  --visualize

# Point cloud as a rotating GIF
python3 scripts/visualize.py \
  --cloud cloud.ply \
  --output cloud.gif --fps 15 --duration 5

# Interactive point cloud view
python3 scripts/visualize.py \
  --cloud cloud.ply \
  --visualize
```

**Command-line options:**
- `--cloud, -c FILE [FILE ...]`: Point cloud file(s) to render (.pcd, .ply, .obj)
- `--traj, -t FILE [FILE ...]`: Trajectory file(s) in TUM format
- `--output, -o FILE`: Save as a rotating GIF to this path
- `--visualize, -v`: Show an interactive 3D window (default if `--output` is omitted)
- `--cloud-bg`: Render clouds in gray so colored trajectories stand out
- `--traj-downsample N`: Keep every Nth trajectory pose when drawing (default: 5)
- `--fps N`: GIF frames per second (default: 10)
- `--duration N`: GIF rotation duration in seconds (default: 5)
- `--width N` / `--height N`: Output resolution (default: 1024×768)

> At least one of `--cloud` / `--traj` is required, and the two can be combined.
> When both are given, a rotating GIF is framed around the trajectory.

## Customization

### Change Alignment Method

Edit `config/pipeline_config.yaml`:
```yaml
alignment:
  global_method: "FPFH_RANSAC"    # Coarse: slower but more robust
  local_method: "P2PLANE_ICP"     # Fine: P2PLANE_ICP (robust) or P2P_ICP (faster)
```
