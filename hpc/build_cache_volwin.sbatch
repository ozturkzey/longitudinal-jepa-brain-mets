#!/usr/bin/env zsh
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --job-name=build_cache_volwin
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Slice cache of each patient's first two visits (Section 3.2) -> cache/brain_mets_slice_cache_volwin.pt
### Usage: sbatch hpc/build_cache_volwin.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

.venv/bin/python -u brain_mets/build_full_slice_cache.py --normalization volume_window
date
