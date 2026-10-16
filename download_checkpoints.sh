#!/usr/bin/env bash
# Downloads the trained weights of release v1.0 into $STABLEWM_HOME/checkpoints/ and verifies them.
# Usage (from the repository root, after editing env.sh): bash download_checkpoints.sh
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
URL=https://github.com/ozturkzey/longitudinal-jepa-brain-mets/releases/download/v1.0
while read -r sum name; do
  model=${name%%.weights_epoch_*}
  dest="${STABLEWM_HOME}/checkpoints/${model}/weights_epoch_${name##*.weights_epoch_}"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 -o "$dest" "${URL}/${name}"
  echo "${sum}  ${dest}" | sha256sum -c -
done < checkpoints.sha256
