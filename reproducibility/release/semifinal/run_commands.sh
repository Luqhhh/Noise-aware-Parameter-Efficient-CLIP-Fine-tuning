#!/usr/bin/env bash
set -euo pipefail
: "${REPOSITORY_ROOT:?Set repository root}"
: "${RECIPE_MANIFEST:?Set an actually registered recipe manifest}"
cd "$REPOSITORY_ROOT/reproducibility/aegis_f1"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -m aegis_clip.cli.reproduce_stage --manifest "$RECIPE_MANIFEST" --audit-only
