#!/usr/bin/env bash
# Multi-seed training for top models (CNN, U-Net, Transformer, MLP).
# Outputs go to experiments/out/{model}/synthetic_seed{N}/ — originals untouched.
#
# Estimated time: ~6 hours GPU (4 models × 3 seeds × ~30 min each)
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
echo "Multi-Seed Training"
echo "=========================================="

SEEDS=(1 2 3)
SEED_MODELS=(cnn unet transformer mlp)

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${SEED_MODELS[@]}"; do
        TAG="seed${SEED}"
        EXTRA_ARGS=""
        if [ "$MODEL" = "transformer" ]; then
            EXTRA_ARGS="--lr 1e-4"
        fi

        echo ""
        echo "--- $MODEL seed=$SEED ---"
        run_if_missing "$OUT/$MODEL/synthetic_${TAG}/best_model.pt" \
            env EXPERIMENT_SEED="$SEED" \
            python -m experiments.train_baseline --model "$MODEL" \
                $EXTRA_ARGS --tag "$TAG"
    done
done

echo ""
echo "=========================================="
echo "Multi-seed training complete!"
echo "=========================================="
echo ""
echo "Run 'python -m experiments.analyze_seeds' to see results."
