#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
LOG_FILE="${REPO_ROOT}/logs/smoltalk_cot_experiments.log"
mkdir -p "$(dirname -- "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >> "${LOG_FILE}"
}

for config in \
  rapid/configs/gemma3/cot/g3_smoltalk_rand_n_base_cot.yaml \
  rapid/configs/gemma3/cot/g3_smoltalk_online_cot.yaml \
  rapid/configs/gemma3/cot/g3_smoltalk_balanced_rand_cot.yaml \
  rapid/configs/gemma3/cot/g3_smoltalk_balanced_online_cot.yaml
do
  log "STARTED ${config}"
  if rapid run -c "${config}"; then
    log "FINISHED ${config}"
  else
    log "FAILED ${config}"
    exit 1
  fi
done
