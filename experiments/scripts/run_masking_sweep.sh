#!/usr/bin/env bash
# Phase 9: Evaluate all models at multiple masking percentages
# Estimated time: ~40 min (GPU for ML models, CPU for SVI)
set -euo pipefail

echo "=========================================="
echo "Phase 9: Masking Sweep"
echo "=========================================="

python -m experiments.eval_masking_sweep --all

echo ""
echo "=========================================="
echo "Masking sweep complete!"
echo "=========================================="
