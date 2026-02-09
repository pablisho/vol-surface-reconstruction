#!/usr/bin/env bash
# Phase 9: Train all models with no-arbitrage constraints (λ_butterfly=0.1)
# Estimated time: ~75 min GPU
#
# Synthetic: 5 new training runs (Transformer already done via lambda sweep)
# Real: 3 fine-tuning runs with constraints
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
echo "Phase 9: Constraint Training"
echo "=========================================="

# --- Synthetic (λ=0.1) ---
echo ""
echo "--- Synthetic: CNN + λ=0.1 ---"
run_if_missing "$OUT/cnn/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model cnn --lambda-butterfly 0.1 --tag arb01

echo ""
echo "--- Synthetic: U-Net + λ=0.1 ---"
run_if_missing "$OUT/unet/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model unet --lambda-butterfly 0.1 --tag arb01

echo ""
echo "--- Synthetic: MLP + λ=0.1 ---"
run_if_missing "$OUT/mlp/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model mlp --lambda-butterfly 0.1 --tag arb01

echo ""
echo "--- Synthetic: FC VAE + λ=0.1 ---"
run_if_missing "$OUT/vae/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model vae --lambda-butterfly 0.1 --tag arb01

echo ""
echo "--- Synthetic: Conv VAE + λ=0.1 ---"
run_if_missing "$OUT/conv_vae/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model conv_vae --lambda-butterfly 0.1 --tag arb01

echo ""
echo "--- Synthetic: Transformer + λ=0.1 ---"
run_if_missing "$OUT/transformer/synthetic_arb01/best_model.pt" \
    python -m experiments.train_baseline --model transformer --lr 1e-4 --lambda-butterfly 0.1 --tag arb01

# --- Real (fine-tuned with constraints) ---
echo ""
echo "--- Real: CNN fine-tune + λ=0.1 ---"
run_if_missing "$OUT/cnn/real_ft_arb01/best_model.pt" \
    python -m experiments.train_baseline --model cnn \
        --data-dir data/real/generated \
        --lr 1e-5 \
        --lambda-butterfly 0.1 \
        --pretrained "$OUT/cnn/synthetic/best_model.pt" \
        --tag ft_arb01

echo ""
echo "--- Real: U-Net fine-tune + λ=0.1 ---"
run_if_missing "$OUT/unet/real_ft_arb01/best_model.pt" \
    python -m experiments.train_baseline --model unet \
        --data-dir data/real/generated \
        --lr 1e-5 \
        --lambda-butterfly 0.1 \
        --pretrained "$OUT/unet/synthetic/best_model.pt" \
        --tag ft_arb01

echo ""
echo "--- Real: Transformer fine-tune + λ=0.1 ---"
run_if_missing "$OUT/transformer/real_ft_arb01/best_model.pt" \
    python -m experiments.train_baseline --model transformer \
        --data-dir data/real/generated \
        --lr 1e-4 \
        --dropout 0.05 \
        --lambda-butterfly 0.1 \
        --pretrained "$OUT/transformer/synthetic/best_model.pt" \
        --tag ft_arb01

echo ""
echo "=========================================="
echo "Constraint training complete!"
echo "=========================================="
