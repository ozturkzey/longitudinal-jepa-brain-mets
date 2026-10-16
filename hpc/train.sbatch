#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Any training stage (encoder, predictor, decoder) from one config in config/train/.
### Usage: sbatch --export=ALL,CONFIG=<config name> hpc/train.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh


if [[ -z "${CONFIG}" ]]; then
  echo "CONFIG is unset -- pass it via --export=ALL,CONFIG=<config name>." >&2
  exit 1
fi

.venv/bin/python -u train.py --config-name "${CONFIG}" ${=EXTRA_ARGS}
date
