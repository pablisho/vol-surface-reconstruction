#!/bin/bash
set -e

echo "=== Phase 10: Thesis-Ready Figures & Computational Analysis ==="
echo ""

# Step 1: Methodology & data figures (no GPU needed, ~2 min)
echo "--- Step 1/3: Methodology & data figures ---"
python -m experiments.generate_thesis_figures

# Step 2: Benchmark inference speed (GPU, ~5 min)
echo ""
echo "--- Step 2/3: Inference benchmark ---"
python -m experiments.benchmark

# Step 3: Comparison figures including qualitative (GPU, ~10 min)
echo ""
echo "--- Step 3/3: Comparison figures (with recompute) ---"
python -m experiments.compare_models --recompute

echo ""
echo "=== Phase 10 complete! ==="
echo "  Methodology figures: experiments/out/thesis_figures/"
echo "  Comparison figures:  experiments/out/comparison/"
echo ""
echo "Total thesis outputs:"
echo "  $(ls experiments/out/thesis_figures/*.pdf 2>/dev/null | wc -l) methodology PDFs"
echo "  $(ls experiments/out/comparison/*.pdf 2>/dev/null | wc -l) comparison PDFs"
echo "  $(ls experiments/out/comparison/*.tex 2>/dev/null | wc -l) LaTeX tables"
