#!/usr/bin/env bash
# Build script for ngram-prep Singularity/Apptainer container
# Builds an immutable SIF for NYU Torch/Greene with a clean, menu-driven UI.

set -Eeuo pipefail
IFS=$'\n\t'

# --------------------------- Config ---------------------------
NETID="${NETID:-edk202}"
DEFAULT_CONTAINER_DIR="${DEFAULT_CONTAINER_DIR:-/scratch/${NETID}/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-ngram-prep.sif}"
DEF_FILE="${DEF_FILE:-environment.def}"
LOG_DIR="${LOG_DIR:-/tmp}"
LOG_FILE="${LOG_DIR}/sif_build_$(date +%Y%m%d_%H%M%S).log"

# Lima config
LIMA_INSTANCE="${LIMA_INSTANCE:-default}"
LIMA_WORKDIR="${LIMA_WORKDIR:-/tmp/lima-builds}"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLD='\033[1m'; RST='\033[0m'

# ---------------------- Tool detection -----------------------
ON_MAC=false
if [[ "$OSTYPE" == "darwin"* ]]; then
  ON_MAC=true
fi

if command -v singularity >/dev/null 2>&1; then
  SIF="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  SIF="apptainer"
else
  if ! $ON_MAC; then
    echo -e "${RED}Error:${RST} Neither 'singularity' nor 'apptainer' found in PATH."
    exit 127
  fi
  # On Mac, Lima build doesn't require local singularity/apptainer
  SIF="apptainer"  # Set default for messaging
fi

# Check for Lima on Mac
LIMA_AVAILABLE=false
if $ON_MAC && command -v limactl >/dev/null 2>&1; then
  LIMA_AVAILABLE=true
fi

# ---------------------- Helper functions ---------------------
say() { printf "%b\n" "$*"; }
info() { say "${GREEN}${*}${RST}"; }
warn() { say "${YELLOW}${*}${RST}"; }
err() { say "${RED}${*}${RST}" >&2; }

die() { err "$*"; exit 1; }

hr() { printf '%*s\n' 70 '' | tr ' ' -; }

# Trap errors to show tail of log
on_err() {
  local rc=$?
  err "\n[ERROR] Build script failed (rc=$rc). Recent log:"
  tail -n 60 "$LOG_FILE" 2>/dev/null || true
  exit "$rc"
}
trap on_err ERR

# ---------------------- Build methods ------------------------
standard_build() {
  local out="$1" def="$2"
  warn "Starting standard build..."
  "$SIF" build "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    info "✓ Standard build succeeded."
    return 0
  else
    return "$rc"
  fi
}

fakeroot_build() {
  local out="$1" def="$2"
  warn "Starting fakeroot build..."
  "$SIF" build --fakeroot "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    info "✓ Fakeroot build succeeded."
    return 0
  else
    return "$rc"
  fi
}

remote_build() {
  local out="$1" def="$2"
  warn "Starting remote build via Sylabs Cloud..."
  warn "Note: requires a Sylabs token (see: ${BLD}${SIF} remote login${RST})"
  # Save & unset bind vars (can conflict with remote service)
  local old_sb="${SINGULARITY_BINDPATH:-}"; local old_ab="${APPTAINER_BINDPATH:-}"
  unset SINGULARITY_BINDPATH || true
  unset APPTAINER_BINDPATH  || true
  
  "$SIF" build --remote "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  
  # Restore
  [ -n "$old_sb" ] && export SINGULARITY_BINDPATH="$old_sb"
  [ -n "$old_ab" ] && export APPTAINER_BINDPATH="$old_ab"
  
  if [ "$rc" -eq 0 ]; then
    info "✓ Remote build succeeded."
    return 0
  else
    return "$rc"
  fi
}

lima_build() {
  local out="$1" def="$2"
  
  if ! $LIMA_AVAILABLE; then
    err "Lima is not available. Install with: brew install lima"
    return 1
  fi
  
  warn "Starting Lima build (Linux VM on macOS)..."
  
  # Check if Lima instance exists
  if ! limactl list 2>/dev/null | grep -q "^${LIMA_INSTANCE}"; then
    warn "Lima instance '${LIMA_INSTANCE}' does not exist. Creating it..."
    info "This may take a few minutes on first run..."
    limactl start "$LIMA_INSTANCE" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
      err "Failed to create Lima instance."
      warn "Try manually: limactl start ${LIMA_INSTANCE}"
      return 1
    fi
    info "✓ Lima instance created and started."
  # Check if Lima instance is running
  elif ! limactl list | grep -q "^${LIMA_INSTANCE}.*Running"; then
    warn "Lima instance '${LIMA_INSTANCE}' not running. Starting it..."
    limactl start "$LIMA_INSTANCE" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
      err "Failed to start Lima instance."
      return 1
    fi
    info "✓ Lima instance started."
  else
    info "✓ Lima instance '${LIMA_INSTANCE}' is running."
  fi
  
  # Get absolute paths
  local abs_def abs_out_dir out_basename
  abs_def="$(cd "$(dirname "$def")" && pwd)/$(basename "$def")"
  abs_out_dir="$(dirname "$out")"
  out_basename="$(basename "$out")"
  mkdir -p "$abs_out_dir"
  abs_out_dir="$(cd "$abs_out_dir" && pwd)"
  
  info "Definition file: $abs_def"
  info "Output location: ${abs_out_dir}/${out_basename}"
  
  # Check if apptainer is installed in Lima
  warn "Checking for Apptainer in Lima VM..."
  if ! limactl shell "$LIMA_INSTANCE" command -v apptainer >/dev/null 2>&1; then
    warn "Apptainer not found in Lima. Installing..."
    limactl shell "$LIMA_INSTANCE" bash <<'INSTALL_SCRIPT' 2>&1 | tee -a "$LOG_FILE"
set -e
# Install dependencies
sudo apt-get update
sudo apt-get install -y wget

# Download and install Apptainer (adjust version as needed)
APPTAINER_VERSION=1.3.2
wget "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb"
sudo apt-get install -y "./apptainer_${APPTAINER_VERSION}_amd64.deb"
rm "apptainer_${APPTAINER_VERSION}_amd64.deb"
INSTALL_SCRIPT
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
      err "Failed to install Apptainer in Lima."
      return 1
    fi
    info "✓ Apptainer installed in Lima."
  else
    info "✓ Apptainer already available in Lima."
  fi
  
  # Build in Lima VM
  warn "Building SIF in Lima (this may take a while)..."
  limactl shell "$LIMA_INSTANCE" bash <<LIMA_BUILD 2>&1 | tee -a "$LOG_FILE"
set -e
cd "$abs_out_dir"
sudo apptainer build "${out_basename}" "${abs_def}"
LIMA_BUILD
  
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    info "✓ Lima build succeeded."
    info "SIF location: ${abs_out_dir}/${out_basename}"
    return 0
  else
    return "$rc"
  fi
}

auto_build() {
  local out="$1" def="$2"
  
  if $ON_MAC && $LIMA_AVAILABLE; then
    warn "Auto mode on macOS: trying Lima first..."
    if lima_build "$out" "$def"; then
      return 0
    fi
    warn "Lima build failed; trying remote..."
    if remote_build "$out" "$def"; then
      return 0
    fi
    return 1
  fi
  
  warn "Auto mode: trying Standard → Fakeroot → Remote"

  "$SIF" build "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    info "✓ Standard build succeeded."
    return 0
  else
    warn "Standard build failed; trying fakeroot…"
  fi

  "$SIF" build --fakeroot "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    info "✓ Fakeroot build succeeded."
    return 0
  else
    warn "Fakeroot build failed; trying remote…"
  fi

  # Remote: temporarily unset bind vars again
  local old_sb="${SINGULARITY_BINDPATH:-}"; local old_ab="${APPTAINER_BINDPATH:-}"
  unset SINGULARITY_BINDPATH || true
  unset APPTAINER_BINDPATH  || true
  
  "$SIF" build --remote "$out" "$def" 2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  
  # Restore
  [ -n "$old_sb" ] && export SINGULARITY_BINDPATH="$old_sb"
  [ -n "$old_ab" ] && export APPTAINER_BINDPATH="$old_ab"
  
  if [ "$rc" -eq 0 ]; then
    info "✓ Remote build succeeded."
    return 0
  fi

  return 1
}

# ---------------------- UI: directory ------------------------
if $ON_MAC; then
  # On Mac, use a sensible default
  DEFAULT_CONTAINER_DIR="${HOME}/containers"
  warn "Running on macOS. Default container directory: ${BLD}${DEFAULT_CONTAINER_DIR}${RST}"
else
  warn "Default container directory: ${BLD}${DEFAULT_CONTAINER_DIR}${RST}"
fi

read -r -p "Press Enter to use default, or type a different path: " USER_INPUT
if [ -z "${USER_INPUT:-}" ]; then
  CONTAINER_DIR="$DEFAULT_CONTAINER_DIR"
  info "Using default directory: $CONTAINER_DIR"
else
  CONTAINER_DIR="$USER_INPUT"
  info "Using custom directory: $CONTAINER_DIR"
fi

[ -f "$DEF_FILE" ] || die "Definition file not found: $DEF_FILE"

info "Creating container directory (if needed): $CONTAINER_DIR"
mkdir -p "$CONTAINER_DIR"

OUTPUT_PATH="${CONTAINER_DIR%/}/${CONTAINER_NAME}"

hr
info "Definition: $DEF_FILE"
info "Output SIF: $OUTPUT_PATH"
if ! $ON_MAC; then
  info "Runtime   : $SIF"
fi
info "Log file  : $LOG_FILE"
if $ON_MAC; then
  if $LIMA_AVAILABLE; then
    info "Lima      : Available (instance: $LIMA_INSTANCE)"
  else
    warn "Lima      : Not installed (install with: brew install lima)"
  fi
fi
hr

# ---------------------- UI: menu -----------------------------
say "${BLD}Select build method:${RST}"
say "  1) Auto (best method for current platform)"
if ! $ON_MAC; then
  say "  2) Standard (local)"
  say "  3) Fakeroot (local, requires fakeroot access)"
fi
if $ON_MAC && $LIMA_AVAILABLE; then
  say "  2) Lima (local Linux VM - builds native Linux SIF)"
fi
say "  4) Remote (Sylabs Cloud)"
say "  q) Quit"

if $ON_MAC && $LIMA_AVAILABLE; then
  read -r -p "Choice [1/2/4/q]: " CHOICE
elif $ON_MAC; then
  read -r -p "Choice [1/4/q]: " CHOICE
else
  read -r -p "Choice [1/2/3/4/q]: " CHOICE
fi
echo

BUILD_OK=false
case "${CHOICE:-1}" in
  1|"")
    warn "Auto-selected."
    if auto_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    ;;
  2)
    if $ON_MAC && $LIMA_AVAILABLE; then
      if lima_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    elif ! $ON_MAC; then
      if standard_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    else
      die "Invalid choice for current platform."
    fi
    ;;
  3)
    if ! $ON_MAC; then
      if fakeroot_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    else
      die "Invalid choice for macOS."
    fi
    ;;
  4)
    if remote_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    ;;
  q|Q)
    warn "Aborted by user."
    exit 0
    ;;
  *)
    die "Unrecognized choice: $CHOICE"
    ;;
esac

# ---------------------- Post status --------------------------
if "$BUILD_OK"; then
  hr
  info "✓ Container built successfully!"
  info "Location: ${OUTPUT_PATH}"
  echo
  if $ON_MAC; then
    warn "Transfer to HPC cluster:"
    echo "  scp $OUTPUT_PATH ${NETID}@greene.hpc.nyu.edu:~/containers/"
    echo
  fi
  warn "Usage examples (on HPC cluster):"
  echo "  # Run Python script"
  echo "  $SIF run $OUTPUT_PATH script.py"
  echo
  echo "  # Interactive Python"
  echo "  $SIF exec $OUTPUT_PATH python"
  echo
  echo "  # Jupyter notebook"
  echo "  $SIF exec $OUTPUT_PATH jupyter notebook"
  echo
  echo "  # Shell access (with GPU)"
  echo "  $SIF shell --nv $OUTPUT_PATH"
  echo
  warn "Note: Use --nv to enable GPU support."
  hr
else
  hr
  err "✗ All selected build attempts failed."
  warn "Options:"
  if $ON_MAC; then
    if ! $LIMA_AVAILABLE; then
      warn "  1. Install Lima: brew install lima"
    fi
    warn "  2. Remote build: run '${SIF} remote login' and add your Sylabs token."
    warn "     (Get token at cloud.sylabs.io/auth/tokens)"
  else
    warn "  1. On Greene: request fakeroot access (hpc@nyu.edu)."
    warn "  2. Remote build: run '${SIF} remote login' and add your Sylabs token."
    warn "     (Get token at cloud.sylabs.io/auth/tokens)"
    warn "  3. On Torch: standard build should work when maintenance is complete."
  fi
  hr
  exit 1
fi
