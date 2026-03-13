#!/bin/bash
# setup_hunspell.sh
# Downloads Hunspell dictionaries for all lexichron languages and configures
# pyenchant to find them within the active conda environment.
#
# Usage:
#   conda activate lexichron
#   bash setup_hunspell.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: must be run inside a conda environment
# ---------------------------------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: no conda environment is active."
    echo "  conda activate lexichron"
    exit 1
fi

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
# enchant searches $CONDA_PREFIX/share/hunspell by default on Linux.
DICT_DIR="$CONDA_PREFIX/share/hunspell"
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$DICT_DIR" "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

echo "Setting up Hunspell dictionaries for lexichron..."
echo "Dictionary directory: $DICT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helper: download a single dictionary pair with error handling
#   download_dict LOCALE AFF_URL DIC_URL
# ---------------------------------------------------------------------------
download_dict() {
    local locale="$1"
    local aff_url="$2"
    local dic_url="$3"

    if [ -f "$DICT_DIR/${locale}.dic" ]; then
        echo "  ✓ ${locale} already present, skipping"
        return 0
    fi

    echo "  Downloading ${locale}..."
    local failed=0

    curl -fsSL -o "$DICT_DIR/${locale}.aff" "$aff_url" || failed=1
    curl -fsSL -o "$DICT_DIR/${locale}.dic" "$dic_url" || failed=1

    if [ $failed -ne 0 ]; then
        echo "  ✗ Failed to download ${locale} — skipping (check URL or network)"
        rm -f "$DICT_DIR/${locale}.aff" "$DICT_DIR/${locale}.dic"
    else
        echo "  ✓ ${locale} installed"
    fi
}

# ---------------------------------------------------------------------------
# Base URL — LibreOffice dictionaries mirror on GitHub (more stable than
# cgit.freedesktop.org, which has historically gone down)
# ---------------------------------------------------------------------------
BASE="https://raw.githubusercontent.com/LibreOffice/dictionaries/master"

# ---------------------------------------------------------------------------
# English variants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Other languages
# ---------------------------------------------------------------------------
echo ""
echo "Other languages:"

# Russian
download_dict ru_RU \
    "$BASE/ru_RU/ru_RU.aff" \
    "$BASE/ru_RU/ru_RU.dic"

# French
download_dict fr_FR \
    "$BASE/fr_FR/fr.aff" \
    "$BASE/fr_FR/fr.dic"

# German (frami variant — comprehensive modern German)
download_dict de_DE \
    "$BASE/de/de_DE_frami.aff" \
    "$BASE/de/de_DE_frami.dic"

download_dict de_AT \
    "$BASE/de/de_AT_frami.aff" \
    "$BASE/de/de_AT_frami.dic"

download_dict de_CH \
    "$BASE/de/de_CH_frami.aff" \
    "$BASE/de/de_CH_frami.dic"

# Spanish
download_dict es_ES \
    "$BASE/es/es_ES.aff" \
    "$BASE/es/es_ES.dic"

download_dict es_MX \
    "$BASE/es/es_MX.aff" \
    "$BASE/es/es_MX.dic"

# Italian
download_dict it_IT \
    "$BASE/it_IT/it_IT.aff" \
    "$BASE/it_IT/it_IT.dic"

# Hebrew
download_dict he_IL \
    "$BASE/he_IL/he_IL.aff" \
    "$BASE/he_IL/he_IL.dic"

# Portuguese
download_dict pt_PT \
    "$BASE/pt_PT/pt_PT.aff" \
    "$BASE/pt_PT/pt_PT.dic"

download_dict pt_BR \
    "$BASE/pt_BR/pt_BR.aff" \
    "$BASE/pt_BR/pt_BR.dic"

# Polish
download_dict pl_PL \
    "$BASE/pl_PL/pl_PL.aff" \
    "$BASE/pl_PL/pl_PL.dic"

# Dutch
download_dict nl_NL \
    "$BASE/nl_NL/nl_NL.aff" \
    "$BASE/nl_NL/nl_NL.dic"

# ---------------------------------------------------------------------------
# Conda activation / deactivation hooks
# Sets PYENCHANT_LIBRARY_PATH so pyenchant finds the conda-installed
# libenchant, and DICPATH so enchant finds the dictionaries in DICT_DIR.
# ---------------------------------------------------------------------------
echo ""
echo "Configuring conda activation scripts..."

cat > "$ACTIVATE_DIR/lexichron_enchant.sh" << 'EOF'
#!/bin/sh
export PYENCHANT_LIBRARY_PATH="$CONDA_PREFIX/lib/libenchant-2.so.2"
export DICPATH="$CONDA_PREFIX/share/hunspell"
EOF
chmod +x "$ACTIVATE_DIR/lexichron_enchant.sh"

cat > "$DEACTIVATE_DIR/lexichron_enchant.sh" << 'EOF'
#!/bin/sh
unset PYENCHANT_LIBRARY_PATH
unset DICPATH
EOF
chmod +x "$DEACTIVATE_DIR/lexichron_enchant.sh"

echo "  ✓ Activation scripts written"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Done. Dictionaries installed in: $DICT_DIR"
echo ""
echo "Installed locales:"
for dic in "$DICT_DIR"/*.dic; do
    [ -f "$dic" ] && echo "  - $(basename "$dic" .dic)"
done
echo ""
echo "Verify pyenchant can see them after reactivating:"
echo "  conda deactivate && conda activate lexichron"
echo "  python -c \"import enchant; print(enchant.list_languages())\""