#!/bin/bash
# Setup hunspell dictionaries for pyenchant
# This script downloads hunspell dictionary files for languages not available via conda

set -e

DICT_DIR="$CONDA_PREFIX/share/hunspell_dictionaries"
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

echo "Setting up hunspell dictionaries for lexichron..."

# Check if conda environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment is active. Please activate the lexichron environment first:"
    echo "  conda activate lexichron"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DICT_DIR"
mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

echo "Dictionary directory: $DICT_DIR"

# Download dictionaries that aren't available via conda
# Russian
if [ ! -f "$DICT_DIR/ru_RU.dic" ]; then
    echo "Downloading Russian dictionary..."
    curl -L -o "$DICT_DIR/ru_RU.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/ru_RU/ru_RU.aff
    curl -L -o "$DICT_DIR/ru_RU.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/ru_RU/ru_RU.dic
    echo "✓ Russian dictionary installed"
else
    echo "✓ Russian dictionary already present"
fi

# French (if not already installed via conda)
if [ ! -f "$DICT_DIR/fr_FR.dic" ] && [ ! -f "$DICT_DIR/fr.dic" ]; then
    echo "Downloading French dictionary..."
    curl -L -o "$DICT_DIR/fr_FR.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/fr_FR/fr.aff
    curl -L -o "$DICT_DIR/fr_FR.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/fr_FR/fr.dic
    echo "✓ French dictionary installed"
else
    echo "✓ French dictionary already present"
fi

# German
if [ ! -f "$DICT_DIR/de_DE.dic" ] && [ ! -f "$DICT_DIR/de.dic" ]; then
    echo "Downloading German dictionary..."
    curl -L -o "$DICT_DIR/de_DE.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/de/de_DE_frami.aff
    curl -L -o "$DICT_DIR/de_DE.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/de/de_DE_frami.dic
    echo "✓ German dictionary installed"
else
    echo "✓ German dictionary already present"
fi

# Spanish
if [ ! -f "$DICT_DIR/es_ES.dic" ] && [ ! -f "$DICT_DIR/es.dic" ]; then
    echo "Downloading Spanish dictionary..."
    curl -L -o "$DICT_DIR/es_ES.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ES.aff
    curl -L -o "$DICT_DIR/es_ES.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ES.dic
    echo "✓ Spanish dictionary installed"
else
    echo "✓ Spanish dictionary already present"
fi

# Italian
if [ ! -f "$DICT_DIR/it_IT.dic" ] && [ ! -f "$DICT_DIR/it.dic" ]; then
    echo "Downloading Italian dictionary..."
    curl -L -o "$DICT_DIR/it_IT.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/it_IT/it_IT.aff
    curl -L -o "$DICT_DIR/it_IT.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/it_IT/it_IT.dic
    echo "✓ Italian dictionary installed"
else
    echo "✓ Italian dictionary already present"
fi

# Hebrew
if [ ! -f "$DICT_DIR/he_IL.dic" ] && [ ! -f "$DICT_DIR/he.dic" ]; then
    echo "Downloading Hebrew dictionary..."
    curl -L -o "$DICT_DIR/he_IL.aff" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/he_IL/he_IL.aff
    curl -L -o "$DICT_DIR/he_IL.dic" https://cgit.freedesktop.org/libreoffice/dictionaries/plain/he_IL/he_IL.dic
    echo "✓ Hebrew dictionary installed"
else
    echo "✓ Hebrew dictionary already present"
fi

# Set up conda activation/deactivation scripts for pyenchant
echo "Setting up conda activation scripts..."

cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/sh
export PYENCHANT_LIBRARY_PATH="$CONDA_PREFIX/lib/libenchant-2.so.2"
EOF
chmod +x "$ACTIVATE_DIR/env_vars.sh"

cat > "$DEACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/sh
unset PYENCHANT_LIBRARY_PATH
EOF
chmod +x "$DEACTIVATE_DIR/env_vars.sh"

echo "✓ Conda activation scripts configured"

echo ""
echo "Setup complete! Hunspell dictionaries installed for:"
echo "  - English (en_US, en_GB, en_CA, en_AU, en_ZA)"
echo "  - Russian (ru_RU)"
echo "  - French (fr_FR)"
echo "  - German (de_DE)"
echo "  - Spanish (es_ES)"
echo "  - Italian (it_IT)"
echo "  - Hebrew (he_IL)"
echo ""
echo "The PYENCHANT_LIBRARY_PATH will be set automatically when you activate the environment."
echo ""
echo "Please deactivate and reactivate your conda environment for changes to take effect:"
echo "  conda deactivate"
echo "  conda activate lexichron"
