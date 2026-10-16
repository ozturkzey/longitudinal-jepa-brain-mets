#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --job-name=build_all_visits_cache
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Slice cache of every visit (Section 3.7.2) -> cache/brain_mets_slice_cache_volwin_all.pt
### Usage: sbatch hpc/build_all_visits_cache.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh
.venv/bin/python brain_mets/build_all_visits_slice_cache.py
date
