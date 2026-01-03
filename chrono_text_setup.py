# Setup - run once per environment
import subprocess
import sys
import os
from pathlib import Path

# Enchant library path fix (must be before any enchant import)
env_path = Path(sys.executable).parent.parent
os.environ.setdefault('PYENCHANT_LIBRARY_PATH', str(env_path / 'lib' / 'libenchant-2.so'))

# Fix hunspell dictionary path if needed
hunspell_src = env_path / 'share' / 'hunspell_dictionaries'
hunspell_dst = env_path / 'share' / 'hunspell'
if hunspell_src.exists() and not hunspell_dst.exists():
    hunspell_dst.symlink_to(hunspell_src)

# Download spaCy models if missing
SPACY_MODELS = [
    'en_core_web_sm',
    'zh_core_web_sm',
    'fr_core_news_sm',
    'de_core_news_sm',
    'it_core_news_sm',
    'ru_core_news_sm',
    'es_core_news_sm',
]

import spacy
for model in SPACY_MODELS:
    try:
        spacy.load(model)
    except OSError:
        print(f"Downloading {model}...")
        subprocess.run([sys.executable, '-m', 'spacy', 'download', model], check=True)