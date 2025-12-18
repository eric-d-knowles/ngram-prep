#!/bin/bash
#
# Build script for chrono-text Apptainer container
# This script uses sbatch to run on a compute node.
#

set -e

DEFINITION_FILE="environment.def"
OUTPUT_IMAGE="chrono-text.sif"
ACCOUNT_ARG=""

# Prompt for account if needed (e.g., for NYU Torch cluster)
read -p "Enter SLURM account name (press Enter to skip): " ACCOUNT_INPUT
read -p "Enter CPUs: " CPUS_INPUT
read -p "Enter memory in G: " MEMG_INPUT
echo

if [ -n "$ACCOUNT_INPUT" ]; then
    ACCOUNT_ARG="--account=$ACCOUNT_INPUT"
    echo "Using account: $ACCOUNT_INPUT"
fi
if [ -n "$CPUS_INPUT" ]; then
    CPUS_ARG="--cpus-per-task=$CPUS_INPUT"
    echo "Using CPUs: $CPUS_INPUT"
fi
if [ -n "$MEMG_INPUT" ]; then
    MEM_ARG="--mem=${MEMG_INPUT}G"
    echo "Using memory: ${MEMG_INPUT}G"
fi

# Check if definition file exists
if [ ! -f "$DEFINITION_FILE" ]; then
    echo "Error: $DEFINITION_FILE not found"
    exit 1
fi

# Remove existing image if present
if [ -f "$OUTPUT_IMAGE" ]; then
    echo "Removing existing $OUTPUT_IMAGE..."
    rm -f "$OUTPUT_IMAGE"
fi

echo
echo "Submitting Apptainer build job to compute node..."
echo "This may take 15-30 minutes depending on network speed."
echo ""

# Submit as batch job
JOBID=$(sbatch $ACCOUNT_ARG $CPUS_ARG $MEM_ARG --time=01:00:00 \
    --output=apptainer-build-%j.out \
    --job-name=apptainer-build \
    <<EOF | awk '{print $NF}'
#!/bin/bash
set -e
cd $(pwd)
apptainer build --fakeroot "$OUTPUT_IMAGE" "$DEFINITION_FILE"
EOF
)

OUTPUT_FILE="apptainer-build-${JOBID}.out"

echo "Job submitted with ID: $JOBID"
echo "Output file: $OUTPUT_FILE"
echo ""
echo "Waiting for job to start..."

# Wait for output file to be created
while [ ! -f "$OUTPUT_FILE" ]; do
    sleep 1
done

echo "Job started! Showing output:"
echo "=========================================="

# Tail the file while job is running
tail -f "$OUTPUT_FILE" &
TAIL_PID=$!

# Monitor job status
while true; do
    JOB_STATE=$(squeue -j $JOBID -h -o %T 2>/dev/null || echo "COMPLETED")
    if [[ "$JOB_STATE" != "RUNNING" && "$JOB_STATE" != "PENDING" ]]; then
        break
    fi
    sleep 2
done

# Kill the tail process
kill $TAIL_PID 2>/dev/null

# Show any remaining output
cat "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Build complete! Container image: $OUTPUT_IMAGE"
echo ""
echo "Test with:"
echo "  apptainer exec --nv $OUTPUT_IMAGE python -c 'import spacy; print(spacy.__version__)'"
