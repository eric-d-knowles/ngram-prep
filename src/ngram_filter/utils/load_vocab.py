from pathlib import Path
from typing import Set

def load_vocab(file_path: str | Path) -> Set[str]:
    """
    Load unique words from a text file where each line is 'word<whitespace>number'.

    Args:
        file_path: Path to the text file.

    Returns:
        A set of unique words (str).
    """
    vocab = set()
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip blank lines
                word = line.split()[0]
                vocab.add(word)

    return vocab
