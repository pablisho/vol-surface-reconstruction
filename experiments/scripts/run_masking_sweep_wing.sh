#!/usr/bin/env bash
# Evaluate existing checkpoints with structured (random+wing) masking.
# Wing masking drops deep OTM points, random drops additional points.
# This is EVALUATION ONLY — no new training needed.
#
# Outputs go to experiments/out/{model}/synthetic/masking_sweep_random_wing.json
#
# Estimated time: ~40 min
set -euo pipefail

echo "=========================================="
echo "Structured Masking Evaluation (random+wing)"
echo "=========================================="

# Note: default wing_threshold=0.5 drops nothing on our grid (max |log_m|=0.357).
# Use --wing-threshold 0.3 so the 2 deepest OTM strikes are always missing,
# adding ~8% structured missingness on top of the random component.
python -m experiments.eval_masking_sweep --all --mask-type random+wing --wing-threshold 0.3

echo ""
echo "=========================================="
echo "Structured masking evaluation complete!"
echo "=========================================="
