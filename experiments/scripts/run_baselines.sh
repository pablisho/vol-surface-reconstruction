#!/usr/bin/env bash
# Reproduce all synthetic baseline trainings (Phases 3-5, 7).
# Skip runs that already have a best_model.pt (or metrics.json for SVI).
#
# All models: lr=1e-3, patience=30, epochs=200 (except Transformer: lr=1e-4)
# Estimated time: ~2 hours GPU
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
echo "Synthetic Baselines"
echo "=========================================="

# --- Phase 3: MLP, CNN, U-Net ---

echo ""
echo "--- MLP ---"
run_if_missing "$OUT/mlp/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model mlp

echo ""
echo "--- CNN ---"
run_if_missing "$OUT/cnn/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model cnn

echo ""
echo "--- U-Net ---"
run_if_missing "$OUT/unet/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model unet

# --- Phase 4: VAEs ---

echo ""
echo "--- FC VAE (train on complete surfaces, beta=1e-4) ---"
run_if_missing "$OUT/vae/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model vae

echo ""
echo "--- Conv VAE (train on complete surfaces, beta=1e-4) ---"
run_if_missing "$OUT/conv_vae/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model conv_vae

# --- Phase 5: Transformer ---

echo ""
echo "--- Transformer (d_model=64, lr=1e-4) ---"
run_if_missing "$OUT/transformer/synthetic/best_model.pt" \
    python -m experiments.train_baseline --model transformer --lr 1e-4

# --- Phase 7: SVI ---

echo ""
echo "--- SVI (per-slice L-BFGS-B, no GPU needed) ---"
run_if_missing "$OUT/svi/synthetic/metrics.json" \
    python -m experiments.eval_svi

echo ""
echo "=========================================="
echo "Synthetic baselines complete!"
echo "=========================================="
