#!/bin/bash
# setup_nlp_resources.sh
#
# Ensures the Enchant C library is available in the active conda environment,
# downloads Hunspell dictionaries into that environment, configures
# activation hooks so pyenchant can find them, and downloads the NLTK
# Swadesh corpus required for Swadesh-anchored Procrustes alignment.

set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: must be run inside a conda environment
# ---------------------------------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: no conda environment is active."
    echo "Activate a conda environment first:"
    echo "  conda activate <env-name>"
    exit 1
fi

ENV_NAME="${CONDA_DEFAULT_ENV:-unknown}"
echo "Active conda environment: $ENV_NAME"
echo "Conda prefix: $CONDA_PREFIX"
echo ""

# ---------------------------------------------------------------------------
# Required tools
# ---------------------------------------------------------------------------
for cmd in conda curl find; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' not found."
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
# Ensure Enchant C library is installed
# ---------------------------------------------------------------------------
echo "Checking for Enchant C library..."

if ! conda list | awk 'NR>3 {print $1}' | grep -qx enchant; then
    echo "  Enchant package not found. Installing..."
    conda install -y -c conda-forge enchant
else
    echo "  ✓ Enchant already installed"
fi

# ---------------------------------------------------------------------------
# Locate Enchant shared library dynamically
# ---------------------------------------------------------------------------
ENCHANT_LIB="$(find "$CONDA_PREFIX/lib" -maxdepth 1 -type f \( -name 'libenchant-2.so*' -o -name 'libenchant.so*' \) | sort | head -n 1 || true)"

if [ -z "$ENCHANT_LIB" ]; then
    echo "Error: Enchant library not found in $CONDA_PREFIX/lib"
    exit 1
fi

echo "  ✓ Found Enchant library:"
echo "    $ENCHANT_LIB"
echo ""

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
download_dict() {
    local locale="$1"
    local aff_url="$2"
    local dic_url="$3"

    if [ -f "$DICT_DIR/${locale}.dic" ] && [ -f "$DICT_DIR/${locale}.aff" ]; then
        echo "  ✓ ${locale} already installed"
        return
    fi

    echo "  Downloading ${locale}..."

    if curl -fsSL -o "$DICT_DIR/${locale}.aff" "$aff_url" &&
       curl -fsSL -o "$DICT_DIR/${locale}.dic" "$dic_url"; then
        echo "  ✓ ${locale} installed"
    else
        echo "  ✗ Failed to download ${locale}"
        rm -f "$DICT_DIR/${locale}.aff" "$DICT_DIR/${locale}.dic"
    fi
}

BASE="https://raw.githubusercontent.com/LibreOffice/dictionaries/master"

echo "Installing Hunspell dictionaries..."
echo ""

download_dict en_US "$BASE/en/en_US.aff" "$BASE/en/en_US.dic"
download_dict en_GB "$BASE/en/en_GB.aff" "$BASE/en/en_GB.dic"
download_dict ru_RU "$BASE/ru_RU/ru_RU.aff" "$BASE/ru_RU/ru_RU.dic"
download_dict fr_FR "$BASE/fr_FR/fr.aff" "$BASE/fr_FR/fr.dic"
download_dict de_DE "$BASE/de/de_DE_frami.aff" "$BASE/de/de_DE_frami.dic"
download_dict es_ES "$BASE/es/es_ES.aff" "$BASE/es/es_ES.dic"
download_dict it_IT "$BASE/it_IT/it_IT.aff" "$BASE/it_IT/it_IT.dic"
download_dict pt_PT "$BASE/pt_PT/pt_PT.aff" "$BASE/pt_PT/pt_PT.dic"
download_dict pt_BR "$BASE/pt_BR/pt_BR.aff" "$BASE/pt_BR/pt_BR.dic"
download_dict nl_NL "$BASE/nl_NL/nl_NL.aff" "$BASE/nl_NL/nl_NL.dic"

# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------
echo ""
echo "Writing conda activation hooks..."

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

echo "✓ Activation scripts installed"
echo ""

# ---------------------------------------------------------------------------
# NLTK Swadesh corpus
# ---------------------------------------------------------------------------
echo "Checking for NLTK Swadesh corpus..."

if python -c "from nltk.corpus import swadesh; swadesh.words('en')" 2>/dev/null; then
    echo "  ✓ Swadesh corpus already installed"
else
    echo "  Downloading Swadesh corpus..."
    if python -c "import nltk; nltk.download('swadesh', quiet=False)"; then
        echo "  ✓ Swadesh corpus installed"
    else
        echo "  ✗ Failed to download Swadesh corpus"
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "Done."
echo "Reactivate the environment and test:"
echo ""
echo "conda deactivate && conda activate $ENV_NAME"
echo "python -c \"import enchant; print(enchant.list_languages())\""
echo "python -c \"from nltk.corpus import swadesh; print(swadesh.words('en')[:5])\""
