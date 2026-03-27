#!/usr/bin/env bash
# ==========================================================================
# REPRODUCIBILITY SCRIPT — reproduces all thesis experiments.
#
# This is a thin orchestrator that calls individual step scripts.
# You can run any step independently:
#
#   bash experiments/scripts/run_baselines.sh          # Step 1: Synthetic baselines
#   bash experiments/scripts/run_real_finetuning.sh    # Step 2: Real data (scratch + FT + transfer)
#   bash experiments/scripts/run_transfer_eval.sh      # Step 3: Zero-shot transfer eval
#   bash experiments/scripts/run_constraint_training.sh # Step 4: Constraint training (λ=0.1)
#   bash experiments/scripts/run_lambda_sweep.sh       # Step 5: Lambda sweep (Pareto)
#   bash experiments/scripts/run_masking_sweep.sh      # Step 6: Masking sweep (random)
#   bash experiments/scripts/run_figures.sh            # Step 7: Figures + tables + benchmark
#
# Prerequisites:
#   1. python -m experiments.generate_dataset        (synthetic, ~20 min CPU)
#   2. python -m experiments.build_real_dataset       (real SPY data)
#   3. Conda env "msthesis" activated
#
# Estimated total time: ~7-8 hours GPU (first run), minutes if cached.
# All steps use skip-if-exists, so safe to re-run after partial completion.
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "THESIS REPRODUCIBILITY SCRIPT"
echo "=========================================="
echo ""

echo "=== Step 1/7: Synthetic Baselines ==="
bash "$SCRIPT_DIR/run_baselines.sh"

echo ""
echo "=== Step 2/7: Real Data (Scratch + Fine-Tuned + SVI) ==="
bash "$SCRIPT_DIR/run_real_finetuning.sh"

echo ""
echo "=== Step 3/7: Transfer Evaluation (Zero-Shot) ==="
bash "$SCRIPT_DIR/run_transfer_eval.sh"

echo ""
echo "=== Step 4/7: Constraint Training (λ=0.1) ==="
bash "$SCRIPT_DIR/run_constraint_training.sh"

echo ""
echo "=== Step 5/7: Lambda Sweep (Pareto Frontier) ==="
bash "$SCRIPT_DIR/run_lambda_sweep.sh"

echo ""
echo "=== Step 6/7: Masking Sweep (Random) ==="
bash "$SCRIPT_DIR/run_masking_sweep.sh"

echo ""
echo "=== Step 7/7: Figures + Tables + Benchmark ==="
bash "$SCRIPT_DIR/run_figures.sh"

echo ""
echo "=========================================="
echo "REPRODUCIBILITY COMPLETE"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Models & metrics:    experiments/out/{model}/{variant}/"
echo "  Comparison figures:  experiments/out/comparison/"
echo "  Thesis figures:      experiments/out/thesis_figures/"
