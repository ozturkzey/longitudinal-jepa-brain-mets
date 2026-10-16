#!/usr/bin/env zsh
#SBATCH --partition=truhnlab
#SBATCH --account=truhnlab
#SBATCH --cpus-per-task=8
#SBATCH --job-name=register
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

### ANTs SyN registration of one patient's FLAIR visits onto the earliest visit.
### Usage: sbatch --export=ALL,PATIENT_ID=<patient_id> hpc/register.sbatch

date
cd "${SLURM_SUBMIT_DIR}"
source env.sh

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

venv/bin/python brain_mets/register_per_modality.py \
  "${BRAINJEPA_RAW}" \
  "${BRAINJEPA_REGISTERED}" \
  --patient ${PATIENT_ID}
