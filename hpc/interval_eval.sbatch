#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --job-name=fixedanchor
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Response to the interval (Section 3.7.2, Section 4.5), with the shuffled-interval control.
### --config reads every later visit of a patient; --train-config is the predictor's own
### training config, which sets the interval and age normalization.
### Usage: sbatch hpc/interval_eval.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

CKPT_ROOT=${STABLEWM_HOME}/checkpoints

.venv/bin/python -u scripts/diagnostics/prediction/fixed_anchor_eval.py \
  --config brain_mets_predictor_consecutive_volwin_e71 \
  --train-config brain_mets_predictor_volwin_e71 \
  --predictor-ckpt "${CKPT_ROOT}/lewm_brain_mets_predictor_volwin_e71/weights_epoch_5.pt" \
  --out-csv results/diagnostics/prediction/fixed_anchor_first_two_volwin_e71_nulled.csv \
  --fig results/diagnostics/prediction/fixed_anchor_first_two_volwin_e71_nulled.png
date
