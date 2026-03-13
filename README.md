## Installation

### Prerequisites

- Git
- Conda or Miniconda

### Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron
```

**Step 2: Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate lexichron
```

**Step 3: Install Hunspell dictionaries**

Spell-checking requires Hunspell dictionaries for all supported languages. Run the
setup script to download and configure them:

```bash
bash scripts/setup_hunspell.sh
```

The script will tell you when to deactivate and reactivate the environment. Once it
does, run:

```bash
conda deactivate
conda activate lexichron
```

**Step 4: Install the package**

```bash
pip install -e .
```

The `-e` flag installs in editable mode, so changes to the source code are immediately
reflected without reinstalling.

**Step 5: Register the Jupyter kernel**

To use the notebooks in the `notebooks/` directory, register the conda environment as
a Jupyter kernel:

```bash
python -m ipykernel install --user --name=lexichron --display-name="Python (lexichron)"
```

You can then select "Python (lexichron)" as the kernel when launching Jupyter.

### Notes

- **spaCy models** are downloaded automatically on first import.
- **Hunspell dictionaries** are handled by the setup script in Step 3 and are not
  downloaded automatically.
- **`rocks-shim`** (a dependency of lexichron) is distributed as a pre-built Linux
  x86_64 wheel. If you are on macOS or Windows, installation will fail at this step.
  HPC cluster users on Linux are unaffected.
