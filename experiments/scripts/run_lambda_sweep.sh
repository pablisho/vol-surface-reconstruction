#!/usr/bin/env bash
# Phase 9: Train top models across multiple λ_butterfly values
# for Pareto frontier analysis (accuracy vs arbitrage tradeoff).
#
# λ=0 already exists (unconstrained baselines).
# This trains the remaining (model, λ) combinations.
# Skip runs that already have a best_model.pt.
#
# All use patience=30 (default). Transformer uses lr=1e-4.
# Estimated time: ~3 hours GPU (up to 14 training runs)
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
echo "Phase 9: Lambda Sweep (Pareto Frontier)"
echo "=========================================="

LAMBDAS=(0.01 0.05 0.1 0.3 1.0)

for LAM in "${LAMBDAS[@]}"; do
    TAG="arb$(echo $LAM | tr -d '.')"

    echo ""
    echo "--- Transformer λ=${LAM} (tag=${TAG}) ---"
    run_if_missing "$OUT/transformer/synthetic_${TAG}/best_model.pt" \
        python -m experiments.train_baseline --model transformer \
            --lr 1e-4 \
            --lambda-butterfly "$LAM" --tag "$TAG"

    echo ""
    echo "--- U-Net λ=${LAM} (tag=${TAG}) ---"
    run_if_missing "$OUT/unet/synthetic_${TAG}/best_model.pt" \
        python -m experiments.train_baseline --model unet \
            --lambda-butterfly "$LAM" --tag "$TAG"

    echo ""
    echo "--- CNN λ=${LAM} (tag=${TAG}) ---"
    run_if_missing "$OUT/cnn/synthetic_${TAG}/best_model.pt" \
        python -m experiments.train_baseline --model cnn \
            --lambda-butterfly "$LAM" --tag "$TAG"
done

echo ""
echo "=========================================="
echo "Lambda sweep complete!"
echo "=========================================="
