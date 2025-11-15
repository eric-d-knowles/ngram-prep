#!/bin/bash
#SBATCH --account=torch_pr_217_general
#SBATCH --job-name=ngram-kit-build
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=ngram_kit_build_%j.log
#SBATCH --error=ngram_kit_build_%j.log

set -euo pipefail

echo "===== ngram-kit container build job started at $(date) ====="
echo "Running on host: $(hostname)"
echo

# ---------------------------------------------------------------------------
# 1. Choose container command (apptainer vs singularity)
#    Optional override:
#      sbatch --export=CONTAINER_CMD=singularity build_ngram_kit_sif.slurm
# ---------------------------------------------------------------------------
: "${CONTAINER_CMD:=apptainer}"

if ! command -v "$CONTAINER_CMD" >/dev/null 2>&1; then
    echo "Command '$CONTAINER_CMD' not found; checking alternatives..."
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_CMD=singularity
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CMD=apptainer
    else
        echo "ERROR: Neither 'apptainer' nor 'singularity' found in PATH." >&2
        exit 1
    fi
fi

echo "Using container command: $CONTAINER_CMD"
echo

# ---------------------------------------------------------------------------
# 2. Paths / configuration
# ---------------------------------------------------------------------------
DEF_FILE="environment.def"
IMAGE_NAME="ngram-kit.sif"

# *** Your requested output directory ***
OUT_DIR="/scratch/edk202/ngram-kit"
mkdir -p "${OUT_DIR}"

IMAGE_PATH="${OUT_DIR}/${IMAGE_NAME}"

echo "Definition file : ${DEF_FILE}"
echo "Output directory: ${OUT_DIR}"
echo "Output image    : ${IMAGE_PATH}"
echo

# ---------------------------------------------------------------------------
# 3. Cache / tmp dirs for Apptainer/Singularity
# ---------------------------------------------------------------------------
export APPTAINER_CACHEDIR="$HOME/.apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

export APPTAINER_TMPDIR="/scratch/$USER/apptainer-tmp"
mkdir -p "$APPTAINER_TMPDIR"

unset SINGULARITY_CACHEDIR
unset SINGULARITY_TMPDIR

echo "APPTAINER_CACHEDIR: $APPTAINER_CACHEDIR"
echo "APPTAINER_TMPDIR  : $APPTAINER_TMPDIR"
echo

# ---------------------------------------------------------------------------
# 4. Build
# ---------------------------------------------------------------------------
echo "Starting container build at $(date)..."
echo

"$CONTAINER_CMD" build --force "${IMAGE_PATH}" "${DEF_FILE}"

echo
echo "===== Build complete at $(date) ====="
echo "Final image written to: ${IMAGE_PATH}"
