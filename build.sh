#!/bin/bash
#SBATCH --account=torch_pr_217_general
#SBATCH --time=1:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=100

# Disable automatic bind mounts that can cause build failures
export APPTAINER_BIND=""
export SINGULARITY_BIND=""

apptainer build /scratch/edk202/containers/ngram-kit.sif /scratch/edk202/ngram-kit/environment.def
