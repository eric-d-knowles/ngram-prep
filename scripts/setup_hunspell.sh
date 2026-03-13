#!/bin/bash
# setup_enchant_hunspell.sh
#
# Ensures the Enchant C library is available in the active conda environment,
# downloads Hunspell dictionaries into that environment, and configures
# activation hooks so pyenchant can find the dictionaries (and, if needed,
# the Enchant shared library).
#
# Usage:
#   conda activate <your-env>
#   bash setup_enchant_hunspell.sh
#
# Notes:
# - Must be run inside an active conda environment.
# - Assumes conda is available on PATH.
# - Designed for Linux/conda workflows.

set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: must be run inside a conda environment
# ---------------------------------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: no conda environment is active."
    echo "Activate a conda environment first, then rerun:"
    echo "  conda activate <env-name>"
    exit 1
fi

ENV_NAME="${CONDA_DEFAULT_ENV:-unknown}"
echo "Active conda environment: $ENV_NAME"
echo "Conda prefix: $CONDA_PREFIX"
echo ""

# ---------------------------------------------------------------------------
# Guard: required tools
# ---------------------------------------------------------------------------
for cmd in conda curl find; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' not found on PATH."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DICT_DIR="$CONDA_PREFIX/share/hunspell"
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$DICT_DIR" "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

echo "Dictionary directory: $DICT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Ensure Enchant C library is installed in this environment
# ---------------------------------------------------------------------------
echo "Checking for Enchant C library..."

if ! conda list | awk 'NR>3 {print $1}' | grep -qx enchant; then
    echo "  Enchant package not found in this environment."
    echo "  Installing via conda-forge..."
    conda install -y -c conda-forge enchant
else
    echo "  ✓ Enchant package already installed"
fi

# ---------------------------------------------------------------------------
# Locate the actual Enchant shared library
# ---------------------------------------------------------------------------
ENCHANT_LIB="$(find "$CONDA_PREFIX/lib" -maxdepth 1 -type f \( -name 'libenchant-2.so*' -o -name 'libenchant.so*' \) | sort | head -n 1 || true)"

if [ -z "$ENCHANT_LIB" ]; then
    echo ""
    echo "Error: Enchant library was not found under:"
    echo "  $CONDA_PREFIX/lib"
    echo ""
    echo "Try checking the installation with:"
    echo "  conda list | grep enchant"
    exit 1
fi

echo "  ✓ Found Enchant library:"
echo "    $ENCHANT_LIB"
echo ""

# ---------------------------------------------------------------------------
# Helper: download a single dictionary pair with error handling
#   download_dict LOCALE AFF_URL DIC_URL
# ---------------------------------------------------------------------------
download_dict() {
    local locale="$1"
    local aff_url="$2"
    local dic_url="$3"

    if [ -f "$DICT_DIR/${locale}.dic" ] && [ -f "$DICT_DIR/${locale}.aff" ]; then
        echo "  ✓ ${locale} already present, skipping"
        return 0
    fi

    echo "  Downloading ${locale}..."
    local failed=0

    curl -fsSL -o "$DICT_DIR/${locale}.aff" "$aff_url" || failed=1
    curl -fsSL -o "$DICT_DIR/${locale}.dic" "$dic_url" || failed=1

    if [ "$failed" -ne 0 ]; then
        echo "  ✗ Failed to download ${locale} — skipping"
        rm -f "$DICT_DIR/${locale}.aff" "$DICT_DIR/${locale}.dic"
    else
        echo "  ✓ ${locale} installed"
    fi
}

# ---------------------------------------------------------------------------
# Base URL — LibreOffice dictionaries mirror on GitHub
# ---------------------------------------------------------------------------
BASE="https://raw.githubusercontent.com/LibreOffice/dictionaries/master"

# ---------------------------------------------------------------------------
# Download dictionaries
# ---------------------------------------------------------------------------
echo "Setting up Hunspell dictionaries..."
echo ""

echo "English:"
download_dict en_US \
    "$BASE/en/en_US.aff" \
    "$BASE/en/en_US.dic"
download_dict en_GB \
    "$BASE/en/en_GB.aff" \
    "$BASE/en/en_GB.dic"
download_dict en_CA \
    "$BASE/en/en_CA.aff" \
    "$BASE/en/en_CA.dic"
download_dict en_AU \
    "$BASE/en/en_AU.aff" \
    "$BASE/en/en_AU.dic"
download_dict en_ZA \
    "$BASE/en/en_ZA.aff" \
    "$BASE/en/en_ZA.dic"

echo ""
echo "Other languages:"
download_dict ru_RU \
    "$BASE/ru_RU/ru_RU.aff" \
    "$BASE/ru_RU/ru_RU.dic"
download_dict fr_FR \
    "$BASE/fr_FR/fr.aff" \
    "$BASE/fr_FR/fr.dic"
download_dict de_DE \
    "$BASE/de/de_DE_frami.aff" \
    "$BASE/de/de_DE_frami.dic"
download_dict de_AT \
    "$BASE/de/de_AT_frami.aff" \
    "$BASE/de/de_AT_frami.dic"
download_dict de_CH \
    "$BASE/de/de_CH_frami.aff" \
    "$BASE/de/de_CH_frami.dic"
download_dict es_ES \
    "$BASE/es/es_ES.aff" \
    "$BASE/es/es_ES.dic"
download_dict es_MX \
    "$BASE/es/es_MX.aff" \
    "$BASE/es/es_MX.dic"
download_dict it_IT \
    "$BASE/it_IT/it_IT.aff" \
    "$BASE/it_IT/it_IT.dic"
download_dict he_IL \
    "$BASE/he_IL/he_IL.aff" \
    "$BASE/he_IL/he_IL.dic"
download_dict pt_PT \
    "$BASE/pt_PT/pt_PT.aff" \
    "$BASE/pt_PT/pt_PT.dic"
download_dict pt_BR \
    "$BASE/pt_BR/pt_BR.aff" \
    "$BASE/pt_BR/pt_BR.dic"
download_dict pl_PL \
    "$BASE/pl_PL/pl_PL.aff" \
    "$BASE/pl_PL/pl_PL.dic"
download_dict nl_NL \
    "$BASE/nl_NL/nl_NL.aff" \
    "$BASE/nl_NL/nl_NL.dic"

# ---------------------------------------------------------------------------
# Write conda activation / deactivation hooks
# ---------------------------------------------------------------------------
# DICPATH helps enchant/hunspell find dictionaries in the env.
# PYENCHANT_LIBRARY_PATH is set to the actual discovered library path.
# ---------------------------------------------------------------------------
echo ""
echo "Configuring conda activation scripts..."

cat > "$ACTIVATE_DIR/enchant_hunspell.sh" << EOF
#!/bin/sh
export PYENCHANT_LIBRARY_PATH="$ENCHANT_LIB"
export DICPATH="$CONDA_PREFIX/share/hunspell"
EOF
chmod +x "$ACTIVATE_DIR/enchant_hunspell.sh"

cat > "$DEACTIVATE_DIR/enchant_hunspell.sh" << 'EOF'
#!/bin/sh
unset PYENCHANT_LIBRARY_PATH
unset DICPATH
EOF
chmod +x "$DEACTIVATE_DIR/enchant_hunspell.sh"

echo "  ✓ Activation scripts written"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "Done."
echo ""
echo "Installed dictionaries in:"
echo "  $DICT_DIR"
echo ""

echo "Installed locales:"
shopt -s nullglob
for dic in "$DICT_DIR"/*.dic; do
    echo "  - $(basename "$dic" .dic)"
done
shopt -u nullglob

echo ""
echo "Next steps:"
echo "  1. Deactivate and reactivate the environment:"
echo "       conda deactivate && conda activate $ENV_NAME"
echo "  2. Test pyenchant:"
echo "       python -c \"import enchant; print(enchant.list_languages())\""
