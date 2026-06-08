#!/bin/bash
# Build Docker image and run FAST-LIO + localization pipeline end-to-end
# Optionally downsamples reference cloud if needed, starts container, and plays rosbag

set -e

# Configuration
IMAGE_NAME="fastlio-localization:latest"
CONTAINER_NAME="fastlio-localization"

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$WORKSPACE_ROOT/config/pipeline_config.yaml"

# Read YAML helper
read_yaml() {
    python3 -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    c = yaml.safe_load(f)
val = c
for k in '$1'.split('.'):
    val = val.get(k)
    if val is None:
        sys.exit(1)
print(val)
"
}

# Input/output paths (relative to workspace)
BAG_PATH_REL="${BAG_PATH_REL:-$(read_yaml data.bag)}"
DURATION="${DURATION:-$(read_yaml processing.duration)}"
REFERENCE_ORIGINAL="${REFERENCE_ORIGINAL:-$(read_yaml data.reference_cloud)}"
REFERENCE_DOWNSAMPLED="${REFERENCE_DOWNSAMPLED:-$(read_yaml data.downsampled_reference_cloud)}"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR_REL="results/run_${TIMESTAMP}"

# Terminal colors for formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Output formatting functions
print_header() { echo ""; echo -e "${BLUE}=== $1 ===${NC}"; echo ""; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Start
print_header "FAST-LIO2 + LOCALIZATION"
echo "Working from: $SCRIPT_DIR"
echo "Workspace root: $WORKSPACE_ROOT"
echo "Output directory: $OUTPUT_DIR_REL"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[!]${NC} Docker not found. Please install Docker Desktop."
    exit 1
fi
print_success "Docker available"

# Build Docker image
cd "$WORKSPACE_ROOT"
print_step "Building Docker image..."
docker build --platform linux/amd64 -t "$IMAGE_NAME" -f docker/Dockerfile .
print_success "Docker image built"

# Verify reference cloud files
ORIGINAL_REF_PATH="$WORKSPACE_ROOT/data/$REFERENCE_ORIGINAL"
DOWNSAMPLED_REF_PATH="$WORKSPACE_ROOT/data/$REFERENCE_DOWNSAMPLED"

if [ ! -f "$ORIGINAL_REF_PATH" ]; then
    echo -e "${YELLOW}[!]${NC} Original reference cloud not found: $ORIGINAL_REF_PATH"
    exit 1
fi

# Create downsampled reference if missing
if [ ! -f "$DOWNSAMPLED_REF_PATH" ]; then
    print_step "Creating downsampled reference cloud (voxel_size=0.1)..."
    docker run --rm \
        --platform linux/amd64 \
        -v "$WORKSPACE_ROOT/data:/data" \
        -v "$WORKSPACE_ROOT/scripts:/opt/fastlio_localization/scripts:ro" \
        "$IMAGE_NAME" \
        python3 /opt/fastlio_localization/scripts/downsample_reference.py \
            --input "/data/$REFERENCE_ORIGINAL" \
            --output "/data/$REFERENCE_DOWNSAMPLED" \
            --voxel 0.1
    if [ $? -eq 0 ] && [ -f "$DOWNSAMPLED_REF_PATH" ]; then
        print_success "Downsampled reference cloud created: $DOWNSAMPLED_REF_PATH"
    else
        echo -e "${YELLOW}[!]${NC} Failed to create downsampled cloud."
        exit 1
    fi
else
    print_info "Using existing downsampled reference cloud: $DOWNSAMPLED_REF_PATH"
fi

# Setup output directory on host
mkdir -p "$WORKSPACE_ROOT/$OUTPUT_DIR_REL"
cp "$WORKSPACE_ROOT/config/pipeline_config.yaml" "$WORKSPACE_ROOT/$OUTPUT_DIR_REL/"

# Remove stale container if exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Start container
print_step "Starting pipeline container..."
docker run -d --platform linux/amd64 \
    --name "$CONTAINER_NAME" \
    -v "$WORKSPACE_ROOT/data:/data:ro" \
    -v "$WORKSPACE_ROOT/$OUTPUT_DIR_REL:/results" \
    -v "$WORKSPACE_ROOT/config:/opt/fastlio_localization/config:ro" \
    -v "$WORKSPACE_ROOT/nodes:/opt/fastlio_localization/nodes:ro" \
    -e REFERENCE_PCD="/data/$REFERENCE_DOWNSAMPLED" \
    -e OUTPUT_DIR="/results" \
    -e ALIGN_INTERVAL="10.0" \
    "$IMAGE_NAME"

print_success "Container started"

# Wait for initialization
echo "Waiting for localization node to initialize (20 seconds)..."
sleep 20

# Play rosbag
INTERNAL_BAG_PATH="/data/$BAG_PATH_REL"
echo "Playing rosbag: $INTERNAL_BAG_PATH"
docker exec "$CONTAINER_NAME" bash -c "
    source /opt/ros/noetic/setup.bash &&
    rosbag play $INTERNAL_BAG_PATH --clock -d 2 --duration $DURATION
"

# Wait for final processing
echo "Waiting for final alignment..."
sleep 15

# Stop container
docker stop "$CONTAINER_NAME"

# Display results
echo ""
print_header "RESULTS"
if [ -d "$WORKSPACE_ROOT/$OUTPUT_DIR_REL" ]; then
    echo "Results saved to: $WORKSPACE_ROOT/$OUTPUT_DIR_REL"
    ls -lh "$WORKSPACE_ROOT/$OUTPUT_DIR_REL"
    if [ -d "$WORKSPACE_ROOT/$OUTPUT_DIR_REL/cached_maps" ]; then
        echo ""
        echo "Cached maps saved in: $WORKSPACE_ROOT/$OUTPUT_DIR_REL/cached_maps"
        ls -lh "$WORKSPACE_ROOT/$OUTPUT_DIR_REL/cached_maps"
    fi
else
    echo -e "${YELLOW}[!]${NC} Output directory not found."
fi