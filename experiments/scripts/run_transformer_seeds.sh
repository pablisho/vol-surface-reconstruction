#!/bin/bash
# Run 2 extra seeds for Transformer lambda sweep (6 lambdas)
# Usage: bash experiments/scripts/run_transformer_seeds.sh
# Runs 2 jobs in parallel at a time

set -e

LAMBDAS=("0:synthetic" "0.01:arb001" "0.05:arb005" "0.1:arb01" "0.5:arb05" "1.0:arb10")
SEEDS=(2 3)

for entry in "${LAMBDAS[@]}"; do
    lam="${entry%%:*}"
    tag="${entry##*:}"

    for seed in "${SEEDS[@]}"; do
        if [ "$tag" = "synthetic" ]; then
            out_tag="s${seed}"
            lam_flag=""
        else
            out_tag="${tag}_s${seed}"
            lam_flag="--lambda-butterfly $lam"
        fi

        out_dir="experiments/out/transformer/synthetic_${out_tag}"
        if [ -d "$out_dir" ] && [ -f "$out_dir/metrics.json" ]; then
            echo "SKIP: $out_dir already exists"
            continue
        fi

        echo "START: transformer lambda=$lam seed=$seed (tag=$out_tag)"
        EXPERIMENT_SEED=$seed python -m experiments.train_baseline \
            --model transformer --lr 1e-4 \
            $lam_flag \
            --tag "$out_tag" &

        # Keep max 2 jobs running
        if (( $(jobs -r | wc -l) >= 2 )); then
            wait -n
        fi
    done
done

wait
echo "All transformer seed runs complete!"
