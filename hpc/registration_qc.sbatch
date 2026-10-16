#!/usr/bin/env zsh
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=regqc
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

### Registration quality control (Section 3.3, Section 4.1), 8 array tasks.
### Usage: sbatch hpc/registration_qc.sbatch
### Merge: awk 'FNR==1 && NR!=1 {next} {print}' results/diagnostics/registration/registration_qc_part*.csv \
###          > results/diagnostics/registration/registration_qc_full.csv

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

CHUNK=158
OFFSET=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

venv/bin/python -u scripts/diagnostics/data/registration_qc.py \
    --offset $OFFSET --n-patients $CHUNK \
    --out results/diagnostics/registration/registration_qc_part${SLURM_ARRAY_TASK_ID}.csv
date
