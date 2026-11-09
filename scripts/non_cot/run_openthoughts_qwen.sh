#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
LOG_FILE="${REPO_ROOT}/logs/openthoughts_experiments.log"
mkdir -p "$(dirname -- "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >> "${LOG_FILE}"
}

for config in \
  rapid/configs/qwen25/qwen25_openthoughts_balanced_rand.yaml \
  rapid/configs/qwen25/qwen25_openthoughts_balanced_online.yaml
do
  log "STARTED ${config}"
  if rapid run -c "${config}"; then
    log "FINISHED ${config}"
  else
    log "FAILED ${config}"
    exit 1
  fi
done
