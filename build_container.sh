#!/bin/bash

# Build script for ngram-prep Singularity container
# This script builds an immutable SIF container for use on NYU's Torch cluster

set -e  # Exit on error

# Configuration
NETID="edk202"
CONTAINER_DIR="/scratch/${NETID}/containers"
CONTAINER_NAME="ngram-prep.sif"
DEF_FILE="environment.def"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo -e "${RED}Error: Definition file '$DEF_FILE' not found${NC}"
    exit 1
fi

# Create container directory if it doesn't exist
echo -e "${GREEN}Creating container directory: ${CONTAINER_DIR}${NC}"
mkdir -p "$CONTAINER_DIR"

# Build the container
echo -e "${GREEN}Building Singularity container...${NC}"
echo -e "${YELLOW}This may take 10-20 minutes${NC}"

OUTPUT_PATH="${CONTAINER_DIR}/${CONTAINER_NAME}"

singularity build "$OUTPUT_PATH" "$DEF_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Container built successfully!${NC}"
    echo -e "${GREEN}Location: ${OUTPUT_PATH}${NC}"
    echo ""
    echo -e "${YELLOW}Usage examples:${NC}"
    echo "  # Run Python script"
    echo "  singularity run $OUTPUT_PATH script.py"
    echo ""
    echo "  # Interactive Python"
    echo "  singularity exec $OUTPUT_PATH python"
    echo ""
    echo "  # Jupyter notebook"
    echo "  singularity exec $OUTPUT_PATH jupyter notebook"
    echo ""
    echo "  # Shell access"
    echo "  singularity shell --nv $OUTPUT_PATH"
    echo ""
    echo -e "${YELLOW}Note: Use --nv flag to enable GPU support${NC}"
else
    echo -e "${RED}✗ Container build failed${NC}"
    exit 1
fi
