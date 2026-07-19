#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:?stage is required}"
export PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
mkdir -p "${ROOT}/artifacts/jobs"
log="${ROOT}/artifacts/jobs/${stage}.log"
exec >>"${log}" 2>&1

echo
echo "stage=${stage}"
echo "started_at=$(date --iso-8601=seconds)"
echo "python=${PYTHON}"
"${ROOT}/scripts/run_stage.sh" "${stage}"
echo "finished_at=$(date --iso-8601=seconds)"
