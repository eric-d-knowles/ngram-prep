#!/bin/bash

# Build script for ngram-prep Singularity container
# This script builds an immutable SIF container for use on NYU's Torch cluster

set -e  # Exit on error

# Configuration
NETID="edk202"
DEFAULT_CONTAINER_DIR="/scratch/${NETID}/containers"
CONTAINER_NAME="ngram-prep.sif"
DEF_FILE="environment.def"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ask user for container directory
echo -e "${YELLOW}Default container directory: ${DEFAULT_CONTAINER_DIR}${NC}"
read -p "Press Enter to use default, or type a different path: " USER_INPUT

if [ -z "$USER_INPUT" ]; then
    CONTAINER_DIR="$DEFAULT_CONTAINER_DIR"
    echo -e "${GREEN}Using default directory: ${CONTAINER_DIR}${NC}"
else
    CONTAINER_DIR="$USER_INPUT"
    echo -e "${GREEN}Using custom directory: ${CONTAINER_DIR}${NC}"
fi

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

# Try different build methods in order of preference
BUILD_SUCCESS=false

# Method 1: Try standard build (works on Torch)
echo -e "${YELLOW}Attempting standard build...${NC}"
if singularity build "$OUTPUT_PATH" "$DEF_FILE" 2>&1 | tee /tmp/build_log.txt | grep -q "FATAL"; then
    echo -e "${YELLOW}Standard build not available${NC}"
else
    if [ -f "$OUTPUT_PATH" ]; then
        BUILD_SUCCESS=true
        echo -e "${GREEN}✓ Standard build succeeded${NC}"
    fi
fi

# Method 2: Try fakeroot (required on Greene with fakeroot access)
if [ "$BUILD_SUCCESS" = false ]; then
    echo -e "${YELLOW}Trying fakeroot build...${NC}"
    if singularity build --fakeroot "$OUTPUT_PATH" "$DEF_FILE" 2>&1; then
        BUILD_SUCCESS=true
        echo -e "${GREEN}✓ Fakeroot build succeeded${NC}"
    else
        echo -e "${YELLOW}Fakeroot build failed${NC}"
    fi
fi

# Method 3: Try remote build (requires Sylabs account)
if [ "$BUILD_SUCCESS" = false ]; then
    echo -e "${YELLOW}Trying remote build via Sylabs Cloud...${NC}"
    echo -e "${YELLOW}Note: This requires a Sylabs account and may take longer${NC}"
    
    # Save and unset bind path variables for remote build
    OLD_SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH}"
    OLD_APPTAINER_BINDPATH="${APPTAINER_BINDPATH}"
    unset SINGULARITY_BINDPATH
    unset APPTAINER_BINDPATH
    
    if singularity build --remote "$OUTPUT_PATH" "$DEF_FILE"; then
        BUILD_SUCCESS=true
        echo -e "${GREEN}✓ Remote build succeeded${NC}"
    else
        echo -e "${RED}Remote build failed${NC}"
    fi
    
    # Restore bind path variables
    export SINGULARITY_BINDPATH="${OLD_SINGULARITY_BINDPATH}"
    export APPTAINER_BINDPATH="${OLD_APPTAINER_BINDPATH}"
fi

# Check final status
if [ "$BUILD_SUCCESS" = true ]; then
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
    echo -e "${RED}✗ All build methods failed${NC}"
    echo -e "${YELLOW}Options to fix:${NC}"
    echo -e "${YELLOW}1. On Greene: Request fakeroot access from hpc@nyu.edu${NC}"
    echo -e "${YELLOW}2. For remote build: Configure Sylabs token with 'singularity remote login'${NC}"
    echo -e "${YELLOW}   Get token from: https://cloud.sylabs.io/auth/tokens${NC}"
    echo -e "${YELLOW}3. On Torch: Standard build should work (when maintenance complete)${NC}"
    exit 1
fi
