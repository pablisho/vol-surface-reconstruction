#!/usr/bin/env bash
# Generate all comparison figures, thesis figures, and benchmarks.
# Estimated time: ~15 min
set -euo pipefail

echo "=========================================="
echo "Generating Figures & Tables"
echo "=========================================="

echo ""
echo "--- Comparison tables and figures ---"
python -m experiments.compare_models --recompute

echo ""
echo "--- Thesis methodology figures ---"
python -m experiments.generate_thesis_figures

echo ""
echo "--- Inference benchmark ---"
python -m experiments.benchmark

echo ""
echo "=========================================="
echo "Figures complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Comparison:  experiments/out/comparison/"
echo "  Thesis:      experiments/out/thesis_figures/"
