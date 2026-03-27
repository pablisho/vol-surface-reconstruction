#!/usr/bin/env bash
# ==========================================================================
# IMPROVED EXPERIMENTS — extensions to strengthen the thesis.
#
# All outputs go to new directories, originals are untouched.
# You can run each improvement independently:
#
#   bash experiments/scripts/run_multi_seed.sh          # ~6h GPU
#   bash experiments/scripts/run_masking_sweep_wing.sh  # ~40 min (eval only)
#   bash experiments/scripts/run_transformer_ablation.sh # ~30 min GPU
#
# Prerequisites: original experiments complete (run_reproduce.sh)
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "IMPROVED EXPERIMENTS"
echo "=========================================="
echo ""

echo "=== Improvement 1/3: Multi-Seed Runs ==="
bash "$SCRIPT_DIR/run_multi_seed.sh"

echo ""
echo "=== Improvement 2/3: Structured Masking (random+wing) ==="
bash "$SCRIPT_DIR/run_masking_sweep_wing.sh"

echo ""
echo "=== Improvement 3/3: Transformer Ablation (No Fourier PE) ==="
bash "$SCRIPT_DIR/run_transformer_ablation.sh"

echo ""
echo "=== Analysis: Figures & Tables for Improved Experiments ==="
bash "$SCRIPT_DIR/run_improved_eval.sh"

echo ""
echo "=========================================="
echo "IMPROVED EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Original results are UNTOUCHED."
