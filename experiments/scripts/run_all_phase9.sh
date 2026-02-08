#!/usr/bin/env bash
# Phase 9: Full reproducible experiment pipeline.
# Runs everything from synthetic baselines to final comparison outputs.
# Each step uses skip-if-exists, so safe to re-run after partial completion.
#
# Estimated total time: ~7 hours GPU (first run), minutes if all cached.
#
# Steps:
#   1. Synthetic baselines (Phases 3-5, 7)        ~2 hours
#   2. Real data fine-tuning (Phase 8)             ~1.5 hours
#   3. Constraint training (λ=0.1, all models)     ~75 min
#   4. Lambda sweep (Pareto frontier, top 3)       ~3 hours
#   5. Masking sweep (eval only)                   ~40 min
#   6. Comparison tables + figures                  ~5 min
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Phase 9: Full experiment suite"
echo ""

echo "=== Step 1/6: Synthetic Baselines ==="
bash "$SCRIPT_DIR/run_baselines.sh"

echo ""
echo "=== Step 2/6: Real Data Fine-Tuning ==="
bash "$SCRIPT_DIR/run_real_finetuning.sh"

echo ""
echo "=== Step 3/6: Constraint Training (λ=0.1) ==="
bash "$SCRIPT_DIR/run_constraint_training.sh"

echo ""
echo "=== Step 4/6: Lambda Sweep (Pareto Frontier) ==="
bash "$SCRIPT_DIR/run_lambda_sweep.sh"

echo ""
echo "=== Step 5/6: Masking Sweep ==="
bash "$SCRIPT_DIR/run_masking_sweep.sh"

echo ""
echo "=== Step 6/6: Generating Comparison Outputs ==="
python -m experiments.compare_models --recompute

echo ""
echo "=========================================="
echo "Phase 9 complete! Outputs in experiments/out/comparison/"
echo "=========================================="
