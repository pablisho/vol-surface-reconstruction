# experiments/benchmark.py
"""Benchmark inference latency, throughput, and GPU memory for all models.

Uses torch.cuda.Event for accurate GPU timing.  SVI is benchmarked on CPU
using time.perf_counter since it doesn't use the GPU.

Usage:
    python -m experiments.benchmark              # all models
    python -m experiments.benchmark --model cnn  # single model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from experiments.train_baseline import build_model

OUT_DIR = Path("experiments/out")
COMPARE_DIR = OUT_DIR / "comparison"

N_WARMUP = 10
N_ITERS = 100


def benchmark_ml_model(
    model_name: str,
    variant: str,
    device: torch.device,
    taus: np.ndarray,
    log_m: np.ndarray,
    sample_input: torch.Tensor,
) -> dict | None:
    """Benchmark a single ML model."""
    ckpt = OUT_DIR / model_name / variant / "best_model.pt"
    if not ckpt.exists():
        print(f"  Checkpoint not found: {ckpt}")
        return None

    n_taus = len(taus)
    n_strikes = len(log_m)

    model = build_model(model_name, n_taus, n_strikes, taus=taus, log_moneyness=log_m)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    inp = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(inp)
    torch.cuda.synchronize()

    # Timed iterations
    torch.cuda.reset_peak_memory_stats(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]

    with torch.no_grad():
        for i in range(N_ITERS):
            start_events[i].record()
            model(inp)
            end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB

    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    throughput = 1000.0 / mean_ms  # surfaces/sec (batch=1)

    return {
        "n_params": n_params,
        "latency_ms": {"mean": round(mean_ms, 2), "std": round(std_ms, 2)},
        "throughput_per_sec": round(throughput, 1),
        "gpu_memory_mb": round(peak_mem, 1),
    }


def benchmark_svi(
    taus: np.ndarray,
    log_m: np.ndarray,
    sample_iv: np.ndarray,
    sample_mask: np.ndarray,
) -> dict:
    """Benchmark SVI per-surface calibration on CPU."""
    from models.svi.calibration import calibrate_surface
    from models.svi.svi import svi_iv

    # Warmup
    for _ in range(3):
        params_list = calibrate_surface(log_m, sample_iv, taus, sample_mask)
        for params, tau in zip(params_list, taus, strict=True):
            svi_iv(log_m, float(tau), params)

    # Timed iterations
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        params_list = calibrate_surface(log_m, sample_iv, taus, sample_mask)
        for params, tau in zip(params_list, taus, strict=True):
            svi_iv(log_m, float(tau), params)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    throughput = 1000.0 / mean_ms

    return {
        "n_params": "40/surf",
        "latency_ms": {"mean": round(mean_ms, 2), "std": round(std_ms, 2)},
        "throughput_per_sec": round(throughput, 1),
        "gpu_memory_mb": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument("--model", type=str, default=None, help="Single model to benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test dataset for sample input
    DATA_DIR = Path("data/synthetic/generated")
    test_ds = VolSurfaceDataset(
        DATA_DIR / "test", mask_config=MaskConfig(mask_type="random", missing_frac=0.3)
    )
    taus = test_ds.taus
    log_m = test_ds.log_moneyness

    # Use first sample as benchmark input (batch=1)
    inp, target, mask, tmsk = test_ds[0]
    sample_input = inp.unsqueeze(0)  # (1, 2, n_taus, n_strikes)

    ml_models = [
        ("Transformer", "transformer", "synthetic"),
        ("CNN", "cnn", "synthetic"),
        ("U-Net", "unet", "synthetic"),
        ("MLP", "mlp", "synthetic"),
        ("FC VAE", "vae", "synthetic"),
        ("Conv VAE", "conv_vae", "synthetic"),
    ]

    if args.model:
        ml_models = [(n, d, v) for n, d, v in ml_models if d == args.model]

    results = {}

    for display, model_name, variant in ml_models:
        print(f"\nBenchmarking {display}...")
        result = benchmark_ml_model(model_name, variant, device, taus, log_m, sample_input)
        if result:
            results[display] = result
            lat = result["latency_ms"]
            print(f"  Latency: {lat['mean']:.2f} +/- {lat['std']:.2f} ms")
            print(f"  Throughput: {result['throughput_per_sec']:.1f} surf/s")
            print(f"  GPU Memory: {result['gpu_memory_mb']:.1f} MB")
            print(f"  Params: {result['n_params']:,}")

    # SVI benchmark
    if args.model is None or args.model == "svi":
        print("\nBenchmarking SVI...")
        masked_iv = inp[0].numpy()
        mask_np = inp[1].numpy().astype(bool)
        result = benchmark_svi(taus, log_m, masked_iv, mask_np)
        results["SVI"] = result
        lat = result["latency_ms"]
        print(f"  Latency: {lat['mean']:.2f} +/- {lat['std']:.2f} ms")
        print(f"  Throughput: {result['throughput_per_sec']:.1f} surf/s")

    # Save results
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARE_DIR / "benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
