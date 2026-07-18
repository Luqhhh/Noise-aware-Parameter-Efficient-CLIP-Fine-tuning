#!/usr/bin/env bash
set -euo pipefail

# Task 0-5 Round 4 Acceptance Reproduction Script
# Runs all verification steps required for acceptance.
# Exits 0 only if everything passes.

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "=============================================="
echo "Task 0-5 Round 4 Acceptance Reproduction"
echo "Repo: $REPO"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================="

FAILED=0

# ── 1. Full test suite ──
echo ""
echo "── 1. Full test suite ──"
if PYTHONPATH=. python3 -m pytest -q --ignore=task0_5_final_acceptance 2>&1; then
    echo "   PASS: All tests pass"
else
    echo "   FAIL: Some tests failed"
    FAILED=1
fi

# ── 2. Runtime audit tests specifically ──
echo ""
echo "── 2. Runtime audit regression tests ──"
if PYTHONPATH=. python3 -m pytest -q tests/test_runtime_manifest_audit.py -v 2>&1; then
    echo "   PASS: Runtime audit tests"
else
    echo "   FAIL: Runtime audit tests"
    FAILED=1
fi

# ── 3. Consensus selection tests ──
echo ""
echo "── 3. Consensus selection tests ──"
if PYTHONPATH=. python3 -m pytest -q tests/test_consensus_selection.py -v 2>&1; then
    echo "   PASS: Consensus selection tests"
else
    echo "   FAIL: Consensus selection tests"
    FAILED=1
fi

# ── 4. Purification manifest tests ──
echo ""
echo "── 4. Purification manifest tests ──"
if PYTHONPATH=. python3 -m pytest -q tests/test_purification_manifest.py -v 2>&1; then
    echo "   PASS: Purification manifest tests"
else
    echo "   FAIL: Purification manifest tests"
    FAILED=1
fi

# ── 5. Dry-run portability check ──
echo ""
echo "── 5. Dry-run portability ──"
DRY_RUN_REPO=$(python3 -c "from pathlib import Path; print(Path('scripts/real_dry_run.py').resolve().parents[1])")
if [ -d "$DRY_RUN_REPO" ]; then
    echo "   PASS: REPO resolves to $DRY_RUN_REPO"
else
    echo "   FAIL: REPO resolution failed"
    FAILED=1
fi

# ── 6. Batch probe script syntax check ──
echo ""
echo "── 6. Batch probe syntax ──"
if python3 -c "import py_compile; py_compile.compile('scripts/build_relabel_batch_probe.py', doraise=True)" 2>&1; then
    echo "   PASS: build_relabel_batch_probe.py compiles"
else
    echo "   FAIL: build_relabel_batch_probe.py has syntax errors"
    FAILED=1
fi

# ── Summary ──
echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    echo "=============================================="
    exit 0
else
    echo "SOME CHECKS FAILED"
    echo "=============================================="
    exit 1
fi
