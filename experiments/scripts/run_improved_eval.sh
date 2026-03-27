#!/usr/bin/env bash
# Analyze all improved experiment results: multi-seed, ablation, structured masking.
# Run this AFTER run_improved.sh (or individual improvement scripts) complete.
#
# Generates:
#   experiments/out/comparison/multi_seed_summary.json
#   experiments/out/comparison/multi_seed_comparison.pdf
#   experiments/out/comparison/table_multi_seed.tex
#   experiments/out/comparison/transformer_ablation.json
#   experiments/out/comparison/masking_comparison.json
#   experiments/out/comparison/masking_comparison.pdf
set -euo pipefail

echo "=========================================="
echo "Improved Experiments — Analysis & Figures"
echo "=========================================="

python -m experiments.analyze_improved

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Outputs in experiments/out/comparison/"
