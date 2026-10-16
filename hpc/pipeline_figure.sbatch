#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pipefig
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Figure 6: real, decoded and predicted later visits for four held-out patients.
### Usage: sbatch hpc/pipeline_figure.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

CKPT_ROOT=${STABLEWM_HOME}/checkpoints

.venv/bin/python -u scripts/diagnostics/decoder/compare_pooled_decode.py \
  --config-name brain_mets_decoder_volwin_e71 \
  --decoder-ckpt "${CKPT_ROOT}/lewm_brain_mets_decoder_volwin_e71/weights_epoch_120.pt" \
  --predictor-ckpt "${CKPT_ROOT}/lewm_brain_mets_predictor_volwin_e71/weights_epoch_5.pt" \
  --residual true \
  --columns 16 --pick 2,13,14,15 \
  --out pipeline_jepa_volwin_firsttwo.png
date
