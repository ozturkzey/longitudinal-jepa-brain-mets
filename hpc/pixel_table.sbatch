#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pixtab
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Pixel-space evaluation (Section 3.7.3): one row of Table 4.3 per submission.
### Usage:
###   sbatch --export=ALL,ARM=jepa      hpc/pixel_table.sbatch
###   sbatch --export=ALL,ARM=radiodino hpc/pixel_table.sbatch
###   sbatch --export=ALL,ARM=mricore   hpc/pixel_table.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

CKPT_ROOT=${STABLEWM_HOME}/checkpoints

case "${ARM}" in
  jepa)
    CONFIG=brain_mets_decoder_volwin_e71
    DECODER=${CKPT_ROOT}/lewm_brain_mets_decoder_volwin_e71/weights_epoch_120.pt
    EXTRA=(--predictor-ckpt "${CKPT_ROOT}/lewm_brain_mets_predictor_volwin_e71/weights_epoch_5.pt"
           --residual true)
    ;;
  radiodino)
    CONFIG=brain_mets_decoder_radiodino_matched
    DECODER=${CKPT_ROOT}/lewm_brain_mets_decoder_radiodino_matched/weights_epoch_80.pt
    EXTRA=(--predictor-ckpt none --residual true)   # projector + predictor come from the config's freeze_encoder_ckpt
    ;;
  mricore)
    CONFIG=brain_mets_decoder_mricore_matched
    DECODER=${CKPT_ROOT}/lewm_brain_mets_decoder_mricore_matched/weights_epoch_95.pt
    EXTRA=(--predictor-ckpt none --residual true)   # projector + predictor come from the config's freeze_encoder_ckpt
    ;;
  *)
    echo "ARM must be jepa, radiodino or mricore (got '${ARM}')" >&2
    exit 1
    ;;
esac

echo "arm=${ARM} config=${CONFIG} decoder=${DECODER}"

.venv/bin/python -u scripts/diagnostics/decoder/compare_pooled_decode.py \
  --config-name "${CONFIG}" \
  --decoder-ckpt "${DECODER}" \
  "${EXTRA[@]}" \
  --score-only
date
