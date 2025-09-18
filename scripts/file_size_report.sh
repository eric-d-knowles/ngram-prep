#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <directory> <pattern>"
    echo "  directory: Path to the directory to search"
    echo "  pattern:   File name pattern (e.g., '*.txt', 'log_*', etc.)"
    echo ""
    echo "Examples:"
    echo "  $0 /home/user/documents '*.pdf'"
    echo "  $0 /var/log 'access_*.log'"
    echo "  $0 . '*.sh'"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required."
    usage
fi

DIRECTORY="$1"
PATTERN="$2"

# Check if directory exists and is accessible
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist or is not accessible."
    exit 1
fi

# Initialize variables
sum=0
sum_squares=0
count=0
sizes=()

# Find files matching the pattern and collect their sizes
while IFS= read -r -d '' file; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null)
        if [ $? -eq 0 ]; then
            sizes+=("$size")
            sum=$((sum + size))
            sum_squares=$((sum_squares + size * size))
            count=$((count + 1))
        fi
    fi
done < <(find "$DIRECTORY" -maxdepth 1 -name "$PATTERN" -print0 2>/dev/null)

# Check if any files were found
if [ $count -eq 0 ]; then
    echo "No files found matching pattern '$PATTERN' in directory '$DIRECTORY'"
    exit 1
fi

# Calculate mean
mean=$(echo "scale=6; $sum / $count" | bc -l)

# Calculate variance using the formula: Var = (sum of squares / n) - (mean^2)
variance=$(echo "scale=6; ($sum_squares / $count) - ($mean * $mean)" | bc -l)

# Calculate standard deviation (square root of variance)
std_dev=$(echo "scale=6; sqrt($variance)" | bc -l)

# Display results
echo "File Size Statistics for pattern '$PATTERN' in '$DIRECTORY':"
echo "=================================================="
echo "Files found: $count"
echo "Total size: $sum bytes"
echo "Mean size: $mean bytes"
echo "Variance: $variance"
echo "Standard deviation: $std_dev bytes"

# Optional: Show individual file sizes for verification (uncomment if needed)
# echo ""
# echo "Individual file sizes:"
# for i in "${!sizes[@]}"; do
#     printf "%d\n" "${sizes[$i]}"
# done