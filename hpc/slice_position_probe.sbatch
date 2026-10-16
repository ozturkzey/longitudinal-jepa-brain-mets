#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=8
#SBATCH --job-name=probes_volwin
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### Slice-position probe (Section 3.6.3): Table 4.1 and Figure 5.
### Usage: sbatch hpc/slice_position_probe.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

CKPT=${STABLEWM_HOME}/checkpoints/lewm_brain_mets_encoder_volwin/weights_epoch_71.pt
CACHE=cache/brain_mets_slice_cache_volwin.pt

.venv/bin/python -u scripts/diagnostics/probes/extract_slice_embeddings.py --ckpt ${CKPT} --tag volwin_e71 --cache ${CACHE}
.venv/bin/python -u scripts/diagnostics/probes/extract_slice_embeddings.py --random-init --tag vit_random_volwin --cache ${CACHE}

.venv/bin/python -u scripts/diagnostics/probes/slice_position_probe.py --tag volwin_e71 --tag vit_random_volwin
date
