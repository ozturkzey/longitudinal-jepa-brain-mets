#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=infobound
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Embedding dimension (Section 3.6.2): Table 4.2. Each --log is that decoder's training
### log (training_logs/); the checkpoint with the lowest held-out reconstruction error is selected from it.
### Usage: sbatch hpc/embedding_dimension.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh


.venv/bin/python -u scripts/diagnostics/decoder/decoder_information_bound.py \
  --arm radiodino768 --log training_logs/decoder_radiodino768_3353852.txt \
  --arm radiodino192 --log training_logs/decoder_radiodino192_3353853.txt
date
