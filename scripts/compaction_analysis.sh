#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <directory> <pattern> [options]"
    echo "  directory: Path to the directory to search"
    echo "  pattern:   File name pattern (e.g., '*.sst', '*.log', etc.)"
    echo ""
    echo "Options:"
    echo "  --rocksdb     Enable RocksDB-specific analysis and recommendations"
    echo "  --bins N      Number of histogram bins (default: 20)"
    echo "  --width N     Maximum histogram bar width (default: 50)"
    echo "  --space       Show total disk space per bin instead of file count"
    echo ""
    echo "Examples:"
    echo "  $0 /data/rocksdb '*.sst' --rocksdb --space"
    echo "  $0 /var/log 'access_*.log'"
    echo "  $0 /data/db '*.sst' --bins 30 --width 60 --space"
    exit 1
}

# Function to convert bytes to human-readable format (awk-only, no bc)
human_readable() {
    local bytes=$1

    awk -v bytes="$bytes" '
    BEGIN {
        units[0] = "B"
        units[1] = "KB"
        units[2] = "MB"
        units[3] = "GB"
        units[4] = "TB"
        units[5] = "PB"

        size = bytes
        unit = 0

        while(size >= 1024 && unit < 5) {
            size = size / 1024
            unit++
        }

        if(unit == 0) {
            printf "%.0f %s", size, units[unit]
        } else {
            printf "%.2f %s", size, units[unit]
        }
    }'
}

# Function to perform all calculations using awk (no bc dependency)
calculate_statistics() {
    local sizes_str="$1"
    local count=$2

    echo "$sizes_str" | awk -v count="$count" '
    BEGIN {
        sum = 0
        sum_sq = 0
        min_size = ""
        max_size = ""
    }
    {
        size = $1
        sum += size
        sum_sq += size * size

        if(min_size == "" || size < min_size) min_size = size
        if(max_size == "" || size > max_size) max_size = size
    }
    END {
        if(count > 0) {
            mean = sum / count
            if(count > 1) {
                variance = (sum_sq / count) - (mean * mean)
                if(variance < 0) variance = 0  # Handle floating point errors
                std_dev = sqrt(variance)
            } else {
                variance = 0
                std_dev = 0
            }
        } else {
            mean = 0
            variance = 0
            std_dev = 0
        }

        # Calculate quartiles
        if(max_size > min_size) {
            range = max_size - min_size
            q1 = min_size + range * 0.25
            q2 = min_size + range * 0.50
            q3 = min_size + range * 0.75
        } else {
            q1 = min_size
            q2 = min_size
            q3 = min_size
        }

        printf "SUM=%.0f\n", sum
        printf "MEAN=%.6f\n", mean
        printf "VARIANCE=%.6f\n", variance
        printf "STDDEV=%.6f\n", std_dev
        printf "MIN=%.0f\n", min_size
        printf "MAX=%.0f\n", max_size
        printf "Q1=%.6f\n", q1
        printf "Q2=%.6f\n", q2
        printf "Q3=%.6f\n", q3
    }'
}

# Function to create histogram efficiently (awk-only, no bc)
create_histogram() {
    local sizes_str="$1"
    local count=$2
    local min_size=$3
    local max_size=$4

    echo "$sizes_str" | awk -v min_size="$min_size" -v max_size="$max_size" -v num_bins="$NUM_BINS" -v show_space="$SHOW_SPACE" '
    BEGIN {
        if(max_size == min_size) {
            print "SINGLE_BIN", min_size, count
            exit
        }

        range = max_size - min_size
        bin_width = range / num_bins

        # Initialize bins
        for(i=0; i<num_bins; i++) {
            bin_count[i] = 0
            bin_space[i] = 0
            bin_start[i] = min_size + i * bin_width
        }
    }
    {
        size = $1
        bin_index = int((size - min_size) / bin_width)
        if(bin_index >= num_bins) bin_index = num_bins - 1
        bin_count[bin_index]++
        bin_space[bin_index] += size
    }
    END {
        max_count = 0
        max_space = 0
        for(i=0; i<num_bins; i++) {
            if(bin_count[i] > max_count) max_count = bin_count[i]
            if(bin_space[i] > max_space) max_space = bin_space[i]
        }

        for(i=0; i<num_bins; i++) {
            bin_end = bin_start[i] + bin_width
            if(show_space == "true") {
                print "BIN", bin_start[i], bin_end, bin_count[i], bin_space[i], max_space
            } else {
                print "BIN", bin_start[i], bin_end, bin_count[i], max_count, "0"
            }
        }
    }'
}

# Function to analyze RocksDB compaction patterns (awk-only, no bc)
analyze_rocksdb_compaction() {
    local count=$1
    local mean=$2
    local std_dev=$3
    local min_size=$4
    local max_size=$5
    local q1=$6
    local q2=$7
    local q3=$8

    echo ""
    echo "RocksDB Compaction Analysis:"
    echo "============================"

    echo "Estimated LSM Tree Structure (approximated):"
    echo "  L0-L1 files (smallest quartile): ~$((count * 25 / 100)) files"
    echo "  L2-L3 files (Q1-Q2): ~$((count * 25 / 100)) files"
    echo "  L4-L5 files (Q2-Q3): ~$((count * 25 / 100)) files"
    echo "  L6+ files (largest quartile): ~$((count * 25 / 100)) files"

    # Calculate metrics using awk
    local metrics=$(awk -v q1="$q1" -v q3="$q3" -v mean="$mean" -v std_dev="$std_dev" '
    BEGIN {
        size_ratio = 1.0
        cv = 0.0

        if(q1 > 0 && q3 > 0) {
            size_ratio = q3 / q1
        }

        if(mean > 0 && std_dev > 0) {
            cv = std_dev / mean * 100
        }

        printf "SIZE_RATIO=%.2f\n", size_ratio
        printf "CV=%.2f\n", cv
    }')

    local size_ratio=$(echo "$metrics" | grep "SIZE_RATIO=" | cut -d= -f2)
    local cv=$(echo "$metrics" | grep "CV=" | cut -d= -f2)

    echo ""
    echo "Compaction Health Indicators:"

    # Check if all files are the same size
    if (( $(awk -v min="$min_size" -v max="$max_size" 'BEGIN{print (min == max ? 1 : 0)}') )); then
        echo "  ✅ All files are identical size - uniform compaction"
        echo "  📊 Perfect size consistency (no variance)"
        return
    fi

    # Analysis based on calculated metrics
    if (( $(awk -v cv="$cv" 'BEGIN{print (cv > 100 ? 1 : 0)}') )); then
        echo "  🔄 HIGH size variance (CV: ${cv}%)"
        echo "     Possible compaction stalls or uneven write patterns"
    elif (( $(awk -v cv="$cv" 'BEGIN{print (cv > 50 ? 1 : 0)}') )); then
        echo "  🔄 Moderate size variance (CV: ${cv}%)"
    else
        echo "  ✅ Low size variance (CV: ${cv}%) - well-balanced compaction"
    fi

    if (( $(awk -v ratio="$size_ratio" 'BEGIN{print (ratio > 15 ? 1 : 0)}') )); then
        echo "  📈 High level amplification: ${size_ratio}x size difference"
        echo "     Consider: Adjust level_compaction_dynamic_level_bytes"
    elif (( $(awk -v ratio="$size_ratio" 'BEGIN{print (ratio > 8 ? 1 : 0)}') )); then
        echo "  📊 Normal level amplification: ${size_ratio}x size difference"
    else
        echo "  📉 Low level amplification: ${size_ratio}x size difference"
    fi

    echo ""
    echo "Optimization Recommendations:"
    if (( $(awk -v cv="$cv" 'BEGIN{print (cv > 80 ? 1 : 0)}') )); then
        echo "  • Check for write spikes causing uneven compaction"
        echo "  • Monitor compaction_readahead_size for sequential reads"
        echo "  • Verify level0_file_num_compaction_trigger isn't too high"
    fi

    if (( $(awk -v ratio="$size_ratio" 'BEGIN{print (ratio > 12 ? 1 : 0)}') )); then
        echo "  • Review target_file_size_base and target_file_size_multiplier"
        echo "  • Consider max_bytes_for_level_multiplier adjustment"
    fi
}

# Parse arguments
DIRECTORY=""
PATTERN=""
ROCKSDB_MODE=false
SHOW_SPACE=false
NUM_BINS=20
MAX_BAR_WIDTH=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --rocksdb)
            ROCKSDB_MODE=true
            shift
            ;;
        --space)
            SHOW_SPACE=true
            shift
            ;;
        --bins)
            NUM_BINS="$2"
            shift 2
            ;;
        --width)
            MAX_BAR_WIDTH="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option $1"
            usage
            ;;
        *)
            if [ -z "$DIRECTORY" ]; then
                DIRECTORY="$1"
            elif [ -z "$PATTERN" ]; then
                PATTERN="$1"
            else
                echo "Too many positional arguments"
                usage
            fi
            shift
            ;;
    esac
done

# Check if required arguments provided
if [ -z "$DIRECTORY" ] || [ -z "$PATTERN" ]; then
    echo "Error: Directory and pattern are required."
    usage
fi

# Check if directory exists and is accessible
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist or is not accessible."
    exit 1
fi

echo "Scanning files in '$DIRECTORY' matching '$PATTERN'..."

# Fast file collection using find and stat in one pass
sizes_output=$(find "$DIRECTORY" -maxdepth 1 -name "$PATTERN" -type f -exec stat -c%s {} \; 2>/dev/null)

if [ -z "$sizes_output" ]; then
    echo "No files found matching pattern '$PATTERN' in directory '$DIRECTORY'"
    exit 1
fi

# Count files and prepare data
count=$(echo "$sizes_output" | wc -l)
echo "Found $count files. Calculating statistics..."

# Calculate all statistics using awk
stats_output=$(calculate_statistics "$sizes_output" "$count")

# Parse results
sum=$(echo "$stats_output" | grep "SUM=" | cut -d= -f2 | tr -d ' ')
mean=$(echo "$stats_output" | grep "MEAN=" | cut -d= -f2 | tr -d ' ')
variance=$(echo "$stats_output" | grep "VARIANCE=" | cut -d= -f2 | tr -d ' ')
std_dev=$(echo "$stats_output" | grep "STDDEV=" | cut -d= -f2 | tr -d ' ')
min_size=$(echo "$stats_output" | grep "MIN=" | cut -d= -f2 | tr -d ' ')
max_size=$(echo "$stats_output" | grep "MAX=" | cut -d= -f2 | tr -d ' ')
q1=$(echo "$stats_output" | grep "Q1=" | cut -d= -f2 | tr -d ' ')
q2=$(echo "$stats_output" | grep "Q2=" | cut -d= -f2 | tr -d ' ')
q3=$(echo "$stats_output" | grep "Q3=" | cut -d= -f2 | tr -d ' ')

# Validate parsed values
if [ -z "$sum" ] || [ -z "$mean" ] || [ -z "$std_dev" ]; then
    echo "Error: Failed to calculate statistics. Raw output:"
    echo "$stats_output"
    exit 1
fi

# Display results
echo ""
echo "File Size Statistics for pattern '$PATTERN' in '$DIRECTORY':"
echo "============================================================="
echo "Files found: $count"
echo "Total size: $(human_readable "$sum") ($sum bytes)"
echo "Mean size: $(human_readable "$mean") ($(printf "%.2f" "$mean") bytes)"
echo "Variance: $(printf "%.2f" "$variance")"
echo "Standard deviation: $(human_readable "$std_dev") ($(printf "%.2f" "$std_dev") bytes)"

if [ $count -gt 1 ]; then
    echo ""
    echo "Additional Statistics:"
    echo "Min size: $(human_readable "$min_size") ($min_size bytes)"
    echo "Max size: $(human_readable "$max_size") ($max_size bytes)"

    # Calculate coefficient of variation using awk
    cv=$(awk -v mean="$mean" -v std_dev="$std_dev" 'BEGIN{
        if(mean > 0) {
            cv = std_dev / mean * 100
            printf "%.2f", cv
        } else {
            printf "0.00"
        }
    }')
    echo "Coefficient of variation: $cv%"
fi

# Create histogram (awk-only, no bc)
echo ""
if [ "$ROCKSDB_MODE" = true ]; then
    echo "SST File Size Distribution (potential LSM levels):"
else
    echo "File Size Histogram:"
fi
echo "================================================="

histogram_output=$(SHOW_SPACE=$SHOW_SPACE create_histogram "$sizes_output" "$count" "$min_size" "$max_size")

if echo "$histogram_output" | grep -q "SINGLE_BIN"; then
    echo "$(human_readable "$min_size") [$count files]: "
    for ((i=0; i<MAX_BAR_WIDTH; i++)); do printf "█"; done
    echo ""
else
    # Process histogram output line by line (awk-only, no bc)
    while IFS= read -r line; do
        if [[ $line == BIN* ]]; then
            # Extract values using awk - now includes space data
            bin_data=$(echo "$line" | awk '{print $2 " " $3 " " $4 " " $5 " " $6}')
            bin_start=$(echo "$bin_data" | awk '{print $1}')
            bin_end=$(echo "$bin_data" | awk '{print $2}')
            bin_count=$(echo "$bin_data" | awk '{print $3}')
            bin_space=$(echo "$bin_data" | awk '{print $4}')
            max_value=$(echo "$bin_data" | awk '{print $5}')

            # Debug: check the values we're getting
            # echo "DEBUG: bin_space=$bin_space, max_value=$max_value" >&2

            # Determine what to show and calculate bar width
            if [ "$SHOW_SPACE" = true ]; then
                display_value="$bin_space"
                # Fix: make sure we're using the right max value for space mode
                if [ "$max_value" != "0" ]; then
                    bar_width=$(awk -v space="$bin_space" -v max="$max_value" -v width="$MAX_BAR_WIDTH" 'BEGIN{
                        if(max > 0) {
                            result = space * width / max
                            if(result < 1 && space > 0) result = 1
                            printf "%.0f", result
                        } else {
                            printf "0"
                        }
                    }')
                else
                    bar_width=0
                fi
            else
                display_value="$bin_count"
                bar_width=$(awk -v count="$bin_count" -v max="$max_value" -v width="$MAX_BAR_WIDTH" 'BEGIN{
                    if(max > 0) {
                        result = count * width / max
                        if(result < 1 && count > 0) result = 1
                        printf "%.0f", result
                    } else {
                        printf "0"
                    }
                }')
            fi

            # Format bin range using standard mixed units
            is_small_range=$(awk -v start="$bin_start" -v end="$bin_end" 'BEGIN{print (start < 1024 && end < 1024 ? 1 : 0)}')

            if [ "$is_small_range" = "1" ]; then
                bin_label=$(awk -v start="$bin_start" -v end="$bin_end" 'BEGIN{printf "%.0f-%.0f B", start, end}')
            else
                bin_start_hr=$(human_readable "$bin_start")
                bin_end_hr=$(human_readable "$bin_end")
                bin_label="${bin_start_hr}-${bin_end_hr}"
            fi

            # Add level hint for RocksDB using awk
            level_hint=""
            if [ "$ROCKSDB_MODE" = true ]; then
                level_hint=$(awk -v start="$bin_start" -v end="$bin_end" -v min="$min_size" -v max="$max_size" '
                BEGIN {
                    middle = (start + end) / 2
                    quartile_size = (max - min) / 4

                    if(middle < min + quartile_size) {
                        print " (L0-L2)"
                    } else if(middle < min + 2 * quartile_size) {
                        print " (L2-L4)"
                    } else if(middle < min + 3 * quartile_size) {
                        print " (L4-L6)"
                    } else {
                        print " (L6+)"
                    }
                }')
            fi

            # Format the display value and label
            if [ "$SHOW_SPACE" = true ]; then
                display_label="$(human_readable "$display_value")"
                printf "%-35s [%12s]: " "$bin_label$level_hint" "$display_label"
            else
                printf "%-35s [%4d]: " "$bin_label$level_hint" "$bin_count"
            fi

            # Draw the bar
            for ((j=0; j<bar_width; j++)); do
                printf "█"
            done
            echo ""
        fi
    done <<< "$histogram_output"

    # Calculate legend using awk
    if [ "$SHOW_SPACE" = true ]; then
        # For space mode, get the max space value from the first line
        max_space=$(echo "$histogram_output" | grep "BIN" | head -1 | awk '{print $6}')
        if [ -n "$max_space" ] && [ "$max_space" != "0" ]; then
            space_per_char=$(awk -v max="$max_space" -v width="$MAX_BAR_WIDTH" 'BEGIN{print max / width}')
            space_per_char_hr=$(human_readable "$space_per_char")
            echo ""
            echo "Legend: Each █ represents ~$space_per_char_hr of disk space"
        fi
    else
        # For count mode, get the max count value
        max_count=$(echo "$histogram_output" | grep "BIN" | head -1 | awk '{print $5}')
        if [ -n "$max_count" ] && [ "$max_count" != "0" ]; then
            files_per_char=$(awk -v max="$max_count" -v width="$MAX_BAR_WIDTH" 'BEGIN{printf "%.1f", max / width}' | sed 's/\.0$//')
            echo ""
            echo "Legend: Each █ represents ~$files_per_char files"
        fi
    fi
fi

# RocksDB-specific analysis
if [ "$ROCKSDB_MODE" = true ]; then
    analyze_rocksdb_compaction "$count" "$mean" "$std_dev" "$min_size" "$max_size" "$q1" "$q2" "$q3"
fi