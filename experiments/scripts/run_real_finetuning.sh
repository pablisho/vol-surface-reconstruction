#!/usr/bin/env bash
# Reproduce all real data fine-tuning experiments (Phase 8).
# Requires synthetic baselines to exist first (run_baselines.sh).
# Skip runs that already have a best_model.pt (or metrics.json for SVI).
#
# Estimated time: ~1.5 hours GPU
set -euo pipefail

OUT="experiments/out"
REAL_DATA="data/real/generated"

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
echo "Real Data Fine-Tuning (Phase 8)"
echo "=========================================="

# --- From-scratch on real data (baselines for transfer learning comparison) ---

echo ""
echo "--- CNN from scratch on real ---"
run_if_missing "$OUT/cnn/real/best_model.pt" \
    python -m experiments.train_baseline --model cnn \
        --data-dir "$REAL_DATA"

echo ""
echo "--- U-Net from scratch on real ---"
run_if_missing "$OUT/unet/real/best_model.pt" \
    python -m experiments.train_baseline --model unet \
        --data-dir "$REAL_DATA"

echo ""
echo "--- MLP from scratch on real ---"
run_if_missing "$OUT/mlp/real/best_model.pt" \
    python -m experiments.train_baseline --model mlp \
        --data-dir "$REAL_DATA"

echo ""
echo "--- Transformer from scratch on real ---"
run_if_missing "$OUT/transformer/real/best_model.pt" \
    python -m experiments.train_baseline --model transformer \
        --data-dir "$REAL_DATA" --lr 1e-4 --patience 30

# --- Fine-tuned from synthetic checkpoints ---

echo ""
echo "--- CNN fine-tuned (lr=1e-5, patience=30) ---"
run_if_missing "$OUT/cnn/real_ft/best_model.pt" \
    python -m experiments.train_baseline --model cnn \
        --data-dir "$REAL_DATA" \
        --lr 1e-5 --patience 30 \
        --pretrained "$OUT/cnn/synthetic/best_model.pt" \
        --tag ft

echo ""
echo "--- U-Net fine-tuned (lr=1e-5, patience=30) ---"
run_if_missing "$OUT/unet/real_ft/best_model.pt" \
    python -m experiments.train_baseline --model unet \
        --data-dir "$REAL_DATA" \
        --lr 1e-5 --patience 30 \
        --pretrained "$OUT/unet/synthetic/best_model.pt" \
        --tag ft

echo ""
echo "--- MLP fine-tuned (lr=1e-5, patience=30) ---"
run_if_missing "$OUT/mlp/real_ft/best_model.pt" \
    python -m experiments.train_baseline --model mlp \
        --data-dir "$REAL_DATA" \
        --lr 1e-5 --patience 30 \
        --pretrained "$OUT/mlp/synthetic/best_model.pt" \
        --tag ft

echo ""
echo "--- Transformer fine-tuned (cosine, dropout=0.05, lr=1e-4) ---"
run_if_missing "$OUT/transformer/real_ft_cosine_d05/best_model.pt" \
    python -m experiments.train_baseline --model transformer \
        --data-dir "$REAL_DATA" \
        --lr 1e-4 --patience 30 --epochs 300 \
        --dropout 0.05 --scheduler cosine \
        --pretrained "$OUT/transformer/synthetic/best_model.pt" \
        --tag ft_cosine_d05

# --- SVI on real data ---

echo ""
echo "--- SVI on real data ---"
run_if_missing "$OUT/svi/real/metrics.json" \
    python -m experiments.eval_svi --data-dir "$REAL_DATA" --tag real

echo ""
echo "=========================================="
echo "Real data fine-tuning complete!"
echo "=========================================="
