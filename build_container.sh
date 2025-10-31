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

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLD='\033[1m'; RST='\033[0m'

# ---------------------- Tool detection -----------------------
if command -v singularity >/dev/null 2>&1; then
  SIF="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  SIF="apptainer"
else
  echo -e "${RED}Error:${RST} Neither 'singularity' nor 'apptainer' found in PATH."
  exit 127
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

auto_build() {
  local out="$1" def="$2"
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
warn "Default container directory: ${BLD}${DEFAULT_CONTAINER_DIR}${RST}"
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
info "Runtime   : $SIF"
info "Log file  : $LOG_FILE"
hr

# ---------------------- UI: menu -----------------------------
say "${BLD}Select build method:${RST}"
say "  1) Auto (Standard → Fakeroot → Remote)"
say "  2) Standard (local)"
say "  3) Fakeroot (local, requires fakeroot access)"
say "  4) Remote (Sylabs Cloud)"
say "  q) Quit"
read -r -p "Choice [1/2/3/4/q]: " CHOICE
echo

BUILD_OK=false
case "${CHOICE:-1}" in
  1|"")
    warn "Auto-selected."
    if auto_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    ;;
  2)
    if standard_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
    ;;
  3)
    if fakeroot_build "$OUTPUT_PATH" "$DEF_FILE"; then BUILD_OK=true; fi
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
  warn "Usage examples:"
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
  warn "  1. On Greene: request fakeroot access (hpc@nyu.edu)."
  warn "  2. Remote build: run '${SIF} remote login' and add your Sylabs token."
  warn "     (Get token at cloud.sylabs.io/auth/tokens)"
  warn "  3. On Torch: standard build should work when maintenance is complete."
  hr
  exit 1
fi
