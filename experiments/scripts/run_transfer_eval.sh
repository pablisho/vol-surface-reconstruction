#!/usr/bin/env bash
# Transfer evaluation: synthetic checkpoint → real test (no fine-tuning)
# Estimated time: ~5 min
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
echo "Transfer Evaluation (Zero-Shot)"
echo "=========================================="

for MODEL in cnn unet mlp transformer; do
    echo ""
    echo "--- Transfer eval: $MODEL (synthetic → real, no FT) ---"
    run_if_missing "$OUT/$MODEL/transfer/metrics.json" \
        python -m experiments.eval_real_transfer --model $MODEL
done

echo ""
echo "=========================================="
echo "Transfer evaluation complete!"
echo "=========================================="
