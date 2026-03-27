#!/usr/bin/env bash
# Transformer ablation: train without Fourier encoding (learnable PE instead).
# Quantifies the contribution of the positional encoding design.
#
# Output: experiments/out/transformer/synthetic_no_fourier/
#
# Estimated time: ~30 min GPU
set -euo pipefail

OUT="experiments/out"

run_if_missing() {
    local marker="$1"
    shift
    if [ -f "$marker" ]; then
        echo "  [SKIP] $marker exists"
    else
        "$@"
    fi
}

echo "=========================================="
echo "Transformer Ablation (No Fourier PE)"
echo "=========================================="

echo ""
echo "--- Transformer WITHOUT Fourier encoding ---"
run_if_missing "$OUT/transformer/synthetic_no_fourier/best_model.pt" \
    python -m experiments.train_baseline --model transformer \
        --lr 1e-4 --no-fourier --tag no_fourier

echo ""
echo "=========================================="
echo "Transformer ablation complete!"
echo "=========================================="
echo ""
echo "Compare with original:"
echo "  Original:    experiments/out/transformer/synthetic/metrics.json"
echo "  No Fourier:  experiments/out/transformer/synthetic_no_fourier/metrics.json"
