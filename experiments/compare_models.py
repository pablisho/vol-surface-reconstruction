# experiments/compare_models.py
"""Generate thesis comparison tables and figures from saved metrics.

Collects experiments/out/{model}/{variant}/metrics.json files,
produces LaTeX-ready tables (booktabs) and publication-quality PDF figures.

Usage:
    python -m experiments.compare_models                # tables + simple figures
    python -m experiments.compare_models --recompute    # full analysis (GPU)
    python -m experiments.compare_models --tables-only
    python -m experiments.compare_models --figures-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT_DIR = Path("experiments/out")
COMPARE_DIR = OUT_DIR / "comparison"

# ---------------------------------------------------------------------------
# Thesis figure styling
# ---------------------------------------------------------------------------

THESIS_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
}

# Model colours (consistent across all figures)
MODEL_COLORS = {
    "U-Net": "#1f77b4",
    "Transformer": "#ff7f0e",
    "CNN": "#2ca02c",
    "MLP": "#9467bd",
    "Conv VAE": "#8c564b",
    "FC VAE": "#e377c2",
    "SVI": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One row in a comparison table."""

    display_name: str
    model: str  # directory name under experiments/out/
    variant: str  # subdirectory
    n_params: int | str


SYNTHETIC_MODELS: list[ModelEntry] = [
    ModelEntry("U-Net", "unet", "synthetic", 265321),
    ModelEntry("Transformer", "transformer", "synthetic", 288129),
    ModelEntry("CNN", "cnn", "synthetic", 295257),
    ModelEntry("MLP", "mlp", "synthetic", 285640),
    ModelEntry("Conv VAE", "conv_vae", "synthetic", 273345),
    ModelEntry("SVI", "svi", "synthetic", "40/surf"),
    ModelEntry("FC VAE", "vae", "synthetic", 284664),
]

REAL_MODELS_SCRATCH: list[ModelEntry] = [
    ModelEntry("CNN", "cnn", "real", 295257),
    ModelEntry("Transformer", "transformer", "real", 288129),
    ModelEntry("U-Net", "unet", "real", 265321),
]

REAL_MODELS_FT: list[ModelEntry] = [
    ModelEntry("CNN (FT)", "cnn", "real_ft", 295257),
    ModelEntry("Transformer (FT)", "transformer", "real_ft_d05", 288129),
    ModelEntry("U-Net (FT)", "unet", "real_ft", 265321),
]

REAL_MODELS_SVI: list[ModelEntry] = [
    ModelEntry("SVI", "svi", "real", "40/surf"),
]

# Combined for backward compat (bar chart etc.)
REAL_MODELS: list[ModelEntry] = REAL_MODELS_FT + REAL_MODELS_SVI

TRANSFER_MODELS: list[ModelEntry] = [
    ModelEntry("U-Net", "unet", "transfer", 265321),
    ModelEntry("CNN", "cnn", "transfer", 295257),
    ModelEntry("Transformer", "transformer", "transfer", 288129),
    ModelEntry("MLP", "mlp", "transfer", 285640),
]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def load_metrics(model: str, variant: str) -> dict | None:
    """Load metrics.json, return None if not found."""
    path = OUT_DIR / model / variant / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_rmse_missing(metrics: dict) -> float | None:
    """Extract RMSE_missing from metrics.json (handles different structures)."""
    for key in ("test_latent_opt", "test_direct", "test"):
        if key in metrics and "rmse_missing" in metrics[key]:
            return metrics[key]["rmse_missing"]
    return None


def extract_test_metrics(metrics: dict) -> dict | None:
    """Extract the test metrics dict."""
    for key in ("test_latent_opt", "test_direct", "test"):
        if key in metrics:
            return metrics[key]
    return None


def extract_butterfly_rate(metrics: dict | None) -> float | None:
    """Extract butterfly violation rate."""
    if metrics is None:
        return None
    arb = metrics.get("arbitrage", {})
    return arb.get("butterfly_rate")


def extract_calendar_rate(metrics: dict | None) -> float | None:
    """Extract calendar violation rate."""
    if metrics is None:
        return None
    arb = metrics.get("arbitrage", {})
    return arb.get("calendar_rate")


def extract_butterfly_severity(metrics: dict | None) -> tuple[float | None, float | None]:
    """Extract butterfly max and mean violation severity."""
    if metrics is None:
        return None, None
    arb = metrics.get("arbitrage", {})
    return arb.get("butterfly_max_violation"), arb.get("butterfly_mean_violation")


def extract_expected_severity(metrics: dict | None) -> float | None:
    """Compute expected butterfly severity per check = rate * mean_violation.

    This combines both frequency and magnitude into a single metric, analogous
    to how RMSE combines frequency and size of reconstruction errors.
    Returns None if either component is missing.
    """
    rate = extract_butterfly_rate(metrics)
    _, mean_sev = extract_butterfly_severity(metrics)
    if rate is None or mean_sev is None:
        return None
    return rate * mean_sev


# ---------------------------------------------------------------------------
# Ground truth arbitrage rates (Heston synthetic, for reference annotations)
# ---------------------------------------------------------------------------


def compute_ground_truth_arbitrage() -> dict[str, float] | None:
    """Compute GT butterfly/calendar rates from the synthetic test set.

    Returns dict with 'butterfly_rate', 'calendar_rate', 'butterfly_expected_severity',
    or None if data not available.
    """
    data_dir = Path("data/synthetic/generated/test")
    if not data_dir.exists():
        return None
    try:
        from data.datasets import VolSurfaceDataset
        from evaluation.arbitrage import surface_arbitrage_report

        ds = VolSurfaceDataset(data_dir)
        cal_total, but_total = 0, 0
        cal_checks, but_checks = 0, 0
        but_sev_sum = 0.0
        for i in range(len(ds)):
            gt_iv = ds.ivs[i]
            report = surface_arbitrage_report(gt_iv, ds.taus, ds.log_moneyness)
            cal_total += report["calendar"]["count"]
            cal_checks += report["calendar"]["total_checks"]
            but_total += report["butterfly"]["count"]
            but_checks += report["butterfly"]["total_checks"]
            if report["butterfly"]["count"] > 0:
                but_sev_sum += report["butterfly"]["mean_violation"] * report["butterfly"]["count"]

        but_rate = but_total / but_checks if but_checks else 0.0
        cal_rate = cal_total / cal_checks if cal_checks else 0.0
        but_mean_sev = but_sev_sum / but_total if but_total else 0.0
        return {
            "butterfly_rate": but_rate,
            "calendar_rate": cal_rate,
            "butterfly_expected_severity": but_rate * but_mean_sev,
        }
    except Exception as e:
        print(f"  Warning: could not compute GT arbitrage: {e}")
        return None


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def _fmt_params(n: int | str) -> str:
    if isinstance(n, str):
        return n
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.1f}\\%"


def _fmt_rmse(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v:.4f}"


def _fmt_sci(v: float | None) -> str:
    if v is None:
        return "--"
    exp = int(np.floor(np.log10(abs(v)))) if v != 0 else 0
    mantissa = v / (10**exp)
    return f"${mantissa:.2f} \\times 10^{{{exp}}}$"


def generate_synthetic_table(entries: list[ModelEntry]) -> str:
    """Booktabs LaTeX table for synthetic results."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params & RMSE$_\text{miss}$ & MAE & Max Error & Calendar & Butterfly \\",
        r"\midrule",
    ]
    for e in entries:
        m = load_metrics(e.model, e.variant)
        if m is None:
            continue
        t = extract_test_metrics(m)
        if t is None:
            continue
        lines.append(
            f"{e.display_name} & {_fmt_params(e.n_params)} & "
            f"{_fmt_rmse(t.get('rmse_missing'))} & "
            f"{_fmt_rmse(t.get('mae'))} & "
            f"{_fmt_rmse(t.get('max_error'))} & "
            f"{_fmt_pct(extract_calendar_rate(m))} & "
            f"{_fmt_pct(extract_butterfly_rate(m))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _real_table_rows(entries: list[ModelEntry]) -> list[str]:
    """Generate table rows for a list of real model entries."""
    rows = []
    for e in entries:
        m = load_metrics(e.model, e.variant)
        if m is None:
            continue
        t = extract_test_metrics(m)
        if t is None:
            continue
        rows.append(
            f"{e.display_name} & "
            f"{_fmt_rmse(t.get('rmse_missing'))} & "
            f"{_fmt_sci(t.get('mse'))} & "
            f"{_fmt_rmse(t.get('mae'))} & "
            f"{_fmt_pct(extract_calendar_rate(m))} & "
            f"{_fmt_pct(extract_butterfly_rate(m))} \\\\"
        )
    return rows


def generate_real_table() -> str:
    """Booktabs LaTeX table for real data: from-scratch, fine-tuned, and SVI."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & RMSE$_\text{miss}$ & Test MSE & MAE & Calendar & Butterfly \\",
    ]

    # From scratch section
    scratch_rows = _real_table_rows(REAL_MODELS_SCRATCH)
    if scratch_rows:
        lines.append(r"\midrule")
        lines.extend(scratch_rows)

    # Fine-tuned section
    ft_rows = _real_table_rows(REAL_MODELS_FT)
    if ft_rows:
        lines.append(r"\midrule")
        lines.extend(ft_rows)

    # SVI section
    svi_rows = _real_table_rows(REAL_MODELS_SVI)
    if svi_rows:
        lines.append(r"\midrule")
        lines.extend(svi_rows)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_arbitrage_table() -> str:
    """Booktabs table comparing unconstrained vs constrained models.

    Uses expected severity (rate * mean_violation) as the severity metric,
    which combines both frequency and magnitude of violations.
    """
    # Pairs: (display, model, unconstrained_variant, constrained_variant)
    pairs = [
        ("Transformer", "transformer", "synthetic", "synthetic_arb01"),
        ("CNN", "cnn", "synthetic", "synthetic_arb01"),
        ("U-Net", "unet", "synthetic", "synthetic_arb01"),
        ("MLP", "mlp", "synthetic", "synthetic_arb01"),
        ("FC VAE", "vae", "synthetic", "synthetic_arb01"),
        ("Conv VAE", "conv_vae", "synthetic", "synthetic_arb01"),
    ]

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{RMSE$_\text{miss}$} & \multicolumn{2}{c}{Butterfly rate}"
        r" & \multicolumn{2}{c}{Expected severity} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Model & $\lambda=0$ & $\lambda=0.1$ & $\lambda=0$ & $\lambda=0.1$"
        r" & $\lambda=0$ & $\lambda=0.1$ \\",
        r"\midrule",
    ]

    for display, model, unc_var, con_var in pairs:
        m_unc = load_metrics(model, unc_var)
        m_con = load_metrics(model, con_var)
        rmse_unc = _fmt_rmse(extract_rmse_missing(m_unc)) if m_unc else "--"
        rmse_con = _fmt_rmse(extract_rmse_missing(m_con)) if m_con else "--"
        but_unc = _fmt_pct(extract_butterfly_rate(m_unc)) if m_unc else "--"
        but_con = _fmt_pct(extract_butterfly_rate(m_con)) if m_con else "--"
        sev_unc = extract_expected_severity(m_unc)
        sev_con = extract_expected_severity(m_con)
        sev_unc_s = _fmt_sci(sev_unc) if sev_unc is not None else "--"
        sev_con_s = _fmt_sci(sev_con) if sev_con is not None else "--"
        lines.append(
            f"{display} & {rmse_unc} & {rmse_con} & {but_unc} & {but_con}"
            f" & {sev_unc_s} & {sev_con_s} \\\\"
        )

    # SVI row
    m_svi = load_metrics("svi", "synthetic")
    if m_svi:
        rmse_svi = _fmt_rmse(extract_rmse_missing(m_svi))
        but_svi = _fmt_pct(extract_butterfly_rate(m_svi))
        sev_svi = extract_expected_severity(m_svi)
        sev_svi_s = _fmt_sci(sev_svi) if sev_svi is not None else "--"
        lines.append(r"\midrule")
        lines.append(f"SVI & {rmse_svi} & -- & {but_svi} & -- & {sev_svi_s} & -- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_rmse_bar_chart() -> matplotlib.figure.Figure:
    """Horizontal bar chart: RMSE_missing for all models on synthetic data."""
    synth_data = []
    for e in SYNTHETIC_MODELS:
        m = load_metrics(e.model, e.variant)
        rmse = extract_rmse_missing(m) if m else None
        synth_data.append((e.display_name, rmse))

    fig, ax = plt.subplots(figsize=(7, 5))

    filtered = [(d[0], d[1]) for d in synth_data if d[1] is not None]
    filtered.sort(key=lambda x: x[1])  # sort by RMSE ascending
    names = [d[0] for d in filtered]
    vals = [d[1] for d in filtered]
    colors = [MODEL_COLORS.get(n, "#333333") for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel(r"RMSE$_\text{missing}$")
    ax.set_title("Synthetic (Heston)")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def fig_rmse_bar_chart_real() -> matplotlib.figure.Figure:
    """Grouped horizontal bar chart: RMSE_missing for real SPY data.

    Shows from-scratch vs fine-tuned side by side, plus SVI.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    real_models = ["CNN", "U-Net", "Transformer"]
    scratch_entries = {e.display_name: e for e in REAL_MODELS_SCRATCH}
    ft_entries = {e.display_name.replace(" (FT)", ""): e for e in REAL_MODELS_FT}

    scratch_vals, ft_vals = [], []
    model_labels = []
    for name in real_models:
        s_entry = scratch_entries.get(name)
        f_entry = ft_entries.get(name)
        s_m = load_metrics(s_entry.model, s_entry.variant) if s_entry else None
        f_m = load_metrics(f_entry.model, f_entry.variant) if f_entry else None
        s_rmse = extract_rmse_missing(s_m) if s_m else 0
        f_rmse = extract_rmse_missing(f_m) if f_m else 0
        scratch_vals.append(s_rmse)
        ft_vals.append(f_rmse)
        model_labels.append(name)

    # SVI
    svi_entry = REAL_MODELS_SVI[0] if REAL_MODELS_SVI else None
    svi_m = load_metrics(svi_entry.model, svi_entry.variant) if svi_entry else None
    svi_rmse = extract_rmse_missing(svi_m) if svi_m else None

    y = np.arange(len(model_labels))
    height = 0.35
    ax.barh(y - height / 2, scratch_vals, height, label="From scratch", color="#5da5da")
    ax.barh(y + height / 2, ft_vals, height, label="Fine-tuned", color="#faa43a")
    if svi_rmse is not None:
        ax.barh(
            len(model_labels),
            svi_rmse,
            height * 2,
            color=MODEL_COLORS.get("SVI", "#7f7f7f"),
            label="SVI",
        )
        model_labels = model_labels + ["SVI"]
        y = np.arange(len(model_labels))

    ax.set_yticks(y)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel(r"RMSE$_\text{missing}$")
    ax.set_title("Real (SPY)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def fig_pareto_accuracy_vs_arbitrage() -> matplotlib.figure.Figure:
    """Scatter: RMSE_missing vs butterfly rate (accuracy-arbitrage tradeoff)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Collect data for both panels
    points = []
    for e in SYNTHETIC_MODELS:
        m = load_metrics(e.model, e.variant)
        if m is None:
            continue
        rmse = extract_rmse_missing(m)
        but = extract_butterfly_rate(m)
        exp_sev = extract_expected_severity(m)
        if rmse is None or but is None:
            continue
        points.append((e.display_name, rmse, but, exp_sev))

    gt = compute_ground_truth_arbitrage()

    for ax, x_fn, x_label, title in [
        (ax1, lambda p: p[2] * 100, "Butterfly violation rate (%)", "Rate"),
        (
            ax2,
            lambda p: (p[3] or 0) * 1e4,
            r"Expected severity ($\times 10^{-4}$)",
            "Expected Severity",
        ),
    ]:
        for name, rmse, but, exp_sev in points:
            x_val = x_fn((name, rmse, but, exp_sev))
            color = MODEL_COLORS.get(name, "#333333")
            ax.scatter(x_val, rmse, s=100, color=color, zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(
                name,
                (x_val, rmse),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
            )

        # Ground truth reference line
        if gt:
            if "Rate" in title:
                gt_x = gt["butterfly_rate"] * 100
            else:
                gt_x = gt["butterfly_expected_severity"] * 1e4
            ax.axvline(
                gt_x,
                color="#aaaaaa",
                linestyle="--",
                linewidth=1,
                zorder=1,
            )
            ax.annotate(
                "GT floor",
                (gt_x, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.007),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=7,
                color="#888888",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(r"RMSE$_\text{missing}$")
        ax.set_title(f"Accuracy vs Arbitrage: {title}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_masking_degradation() -> matplotlib.figure.Figure | None:
    """Line plot: RMSE_missing vs masking % for each model."""
    # Load masking sweep results
    sweep_data = {}
    for model_name in ["mlp", "cnn", "unet", "transformer", "vae", "conv_vae", "svi"]:
        path = OUT_DIR / model_name / "synthetic" / "masking_sweep.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            sweep_data[model_name] = data["masking_levels"]

    if not sweep_data:
        print("  No masking sweep data found, skipping masking_degradation figure")
        return None

    display_names = {
        "mlp": "MLP",
        "cnn": "CNN",
        "unet": "U-Net",
        "transformer": "Transformer",
        "vae": "FC VAE",
        "conv_vae": "Conv VAE",
        "svi": "SVI",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, levels in sweep_data.items():
        fracs = sorted(float(k) for k in levels.keys())
        rmses = [levels[str(f)]["rmse_missing"] for f in fracs]
        display = display_names.get(model_name, model_name)
        color = MODEL_COLORS.get(display, "#333333")
        ax.plot(
            [f * 100 for f in fracs],
            rmses,
            marker="o",
            label=display,
            color=color,
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Missing fraction (%)")
    ax.set_ylabel(r"RMSE$_\text{missing}$")
    ax.set_title("Reconstruction Quality vs Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_transfer_waterfall() -> matplotlib.figure.Figure:
    """Bar chart: from-scratch -> transfer -> fine-tuned for each model."""
    models = [
        ("CNN", "cnn", "real", "transfer", "real_ft"),
        ("U-Net", "unet", "real", "transfer", "real_ft"),
        ("Transformer", "transformer", "real", "transfer", "real_ft_d05"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25

    for i, stage_label, variant_key in [
        (0, "From scratch", 2),
        (1, "Transfer (no FT)", 3),
        (2, "Fine-tuned", 4),
    ]:
        vals = []
        for entry in models:
            m = load_metrics(entry[1], entry[variant_key])
            rmse = extract_rmse_missing(m) if m else 0
            vals.append(rmse)
        ax.bar(x + i * width, vals, width, label=stage_label)

    ax.set_xlabel("Model")
    ax.set_ylabel(r"RMSE$_\text{missing}$")
    ax.set_title("Transfer Learning Impact (Real SPY Data)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m[0] for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_constraint_impact() -> matplotlib.figure.Figure:
    """Grouped bar chart: unconstrained vs constrained — RMSE, violation rate, severity."""
    pairs = [
        ("Transformer", "transformer", "synthetic", "synthetic_arb01"),
        ("CNN", "cnn", "synthetic", "synthetic_arb01"),
        ("U-Net", "unet", "synthetic", "synthetic_arb01"),
        ("MLP", "mlp", "synthetic", "synthetic_arb01"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(pairs))
    width = 0.35

    # RMSE comparison
    rmse_unc, rmse_con = [], []
    for _, model, unc, con in pairs:
        m_u = load_metrics(model, unc)
        m_c = load_metrics(model, con)
        rmse_unc.append(extract_rmse_missing(m_u) if m_u else 0)
        rmse_con.append(extract_rmse_missing(m_c) if m_c else 0)

    ax1.bar(x - width / 2, rmse_unc, width, label=r"$\lambda=0$", color="#1f77b4")
    ax1.bar(x + width / 2, rmse_con, width, label=r"$\lambda=0.1$", color="#d62728")
    ax1.set_ylabel(r"RMSE$_\text{missing}$")
    ax1.set_title("Reconstruction Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[0] for p in pairs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Butterfly rate comparison
    but_unc, but_con = [], []
    for _, model, unc, con in pairs:
        m_u = load_metrics(model, unc)
        m_c = load_metrics(model, con)
        but_unc.append((extract_butterfly_rate(m_u) or 0) * 100)
        but_con.append((extract_butterfly_rate(m_c) or 0) * 100)

    ax2.bar(x - width / 2, but_unc, width, label=r"$\lambda=0$", color="#1f77b4")
    ax2.bar(x + width / 2, but_con, width, label=r"$\lambda=0.1$", color="#d62728")
    ax2.set_ylabel("Butterfly violation rate (%)")
    ax2.set_title("Violation Rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels([p[0] for p in pairs])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Expected severity comparison
    sev_unc, sev_con = [], []
    for _, model, unc, con in pairs:
        m_u = load_metrics(model, unc)
        m_c = load_metrics(model, con)
        sev_unc.append((extract_expected_severity(m_u) or 0) * 1e4)
        sev_con.append((extract_expected_severity(m_c) or 0) * 1e4)

    ax3.bar(x - width / 2, sev_unc, width, label=r"$\lambda=0$", color="#1f77b4")
    ax3.bar(x + width / 2, sev_con, width, label=r"$\lambda=0.1$", color="#d62728")
    ax3.set_ylabel(r"Expected severity ($\times 10^{-4}$)")
    ax3.set_title("Violation Severity")
    ax3.set_xticks(x)
    ax3.set_xticklabels([p[0] for p in pairs])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def _collect_lambda_sweep_data() -> tuple[
    list[tuple[str, str]],
    dict[str, float],
    list[tuple[str, list[tuple[float, float, float, float | None]]]],
]:
    """Collect lambda sweep data for reuse across figure variants.

    Returns (sweep_models, lambda_tags, model_data) where model_data is
    [(display, [(lambda, rmse, butterfly_rate, expected_severity), ...]), ...].
    """
    sweep_models = [
        ("Transformer", "transformer"),
        ("U-Net", "unet"),
        ("CNN", "cnn"),
    ]
    lambda_tags = {
        "synthetic": 0.0,
        "synthetic_arb001": 0.01,
        # "synthetic_arb002": 0.02,
        "synthetic_arb005": 0.05,
        # "synthetic_arb007": 0.07,
        "synthetic_arb01": 0.1,
        # "synthetic_arb015": 0.15,
        # "synthetic_arb02": 0.2,
        # "synthetic_arb03": 0.3,
        "synthetic_arb05": 0.5,
        # "synthetic_arb07": 0.7,
        "synthetic_arb10": 1.0,
    }

    model_data = []
    seed_suffixes = ["", "_s2", "_s3"]
    for display, model_dir in sweep_models:
        points = []
        for variant, lam in lambda_tags.items():
            rmses, buts, sevs = [], [], []
            for suf in seed_suffixes:
                m = load_metrics(model_dir, f"{variant}{suf}")
                if m is None:
                    continue
                r = extract_rmse_missing(m)
                b = extract_butterfly_rate(m)
                s = extract_expected_severity(m)
                if r is not None and b is not None:
                    rmses.append(r)
                    buts.append(b)
                    sevs.append(s if s is not None else 0)
            if rmses:
                points.append(
                    (
                        lam,
                        sum(rmses) / len(rmses),
                        sum(buts) / len(buts),
                        sum(sevs) / len(sevs) if sevs else None,
                    )
                )
        if len(points) >= 2:
            points.sort(key=lambda p: p[0])
            model_data.append((display, points))

    return sweep_models, lambda_tags, model_data


def _annotate_lambda_points(
    ax: matplotlib.axes.Axes,
    xs: list[float],
    ys: list[float],
    lams: list[float],
    color: str,
) -> None:
    """Annotate lambda sweep points — only key values to reduce clutter."""
    # Only label: no reg (first), λ=0.1 (sweet spot), λ=1.0 (last)
    key_lams = {0.0, 0.1, 1.0}
    for i, (x, y, lam) in enumerate(zip(xs, ys, lams, strict=False)):
        if lam not in key_lams:
            continue
        label = f"$\\lambda$={lam}" if lam > 0 else "no reg."
        # First point: right, last point: left, middle: above
        if i == 0:
            xytext = (8, 0)
            ha, va = "left", "center"
        elif i == len(xs) - 1:
            xytext = (-8, 0)
            ha, va = "right", "center"
        else:
            xytext = (0, 8)
            ha, va = "center", "bottom"
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=7,
            color=color,
            ha=ha,
            va=va,
        )


def _pareto_front(xs: list[float], ys: list[float]):
    """Return indices of Pareto-optimal points (lower x and lower y is better)."""
    n = len(xs)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


def fig_pareto_lambda_sweep() -> matplotlib.figure.Figure | None:
    """Lambda sweep: RMSE and butterfly metrics vs λ.

    Panel 1: RMSE vs λ.
    Panel 2: Butterfly violation rate vs λ.
    Panel 3 (if severity available): Expected severity vs λ.
    """
    _, _, model_data = _collect_lambda_sweep_data()

    if not model_data:
        print("  No lambda sweep data found, skipping pareto_lambda_sweep figure")
        return None

    has_severity = any(p[3] is not None for _, points in model_data for p in points)

    if has_severity:
        fig, (ax_rmse, ax_but, ax_sev) = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, (ax_rmse, ax_but) = plt.subplots(1, 2, figsize=(12, 5))
        ax_sev = None

    gt = compute_ground_truth_arbitrage()

    # Collect all unique lambda values across models, map to evenly-spaced positions
    all_lams = sorted({p[0] for _, points in model_data for p in points})
    lam_to_pos = {lam: i for i, lam in enumerate(all_lams)}

    for display, points in model_data:
        positions = [lam_to_pos[p[0]] for p in points]
        rmses = [p[1] for p in points]
        buts = [p[2] * 100 for p in points]
        color = MODEL_COLORS.get(display, "#333333")

        ax_rmse.plot(
            positions, rmses, marker="o", label=display, color=color, linewidth=2, markersize=5
        )
        ax_but.plot(
            positions, buts, marker="o", label=display, color=color, linewidth=2, markersize=5
        )

        if ax_sev is not None:
            sevs = [(p[3] or 0) * 1e4 for p in points]
            ax_sev.plot(
                positions, sevs, marker="o", label=display, color=color, linewidth=2, markersize=5
            )

    # SVI horizontal references
    m_svi = load_metrics("svi", "synthetic")
    if m_svi:
        svi_color = MODEL_COLORS.get("SVI", "#7f7f7f")
        but_svi = extract_butterfly_rate(m_svi)
        rmse_svi = extract_rmse_missing(m_svi)
        exp_sev_svi = extract_expected_severity(m_svi)
        if rmse_svi:
            ax_rmse.axhline(rmse_svi, color=svi_color, linestyle="--", linewidth=1.5, label="SVI")
        if but_svi is not None:
            ax_but.axhline(
                but_svi * 100, color=svi_color, linestyle="--", linewidth=1.5, label="SVI"
            )
        if ax_sev is not None and exp_sev_svi is not None:
            ax_sev.axhline(
                exp_sev_svi * 1e4, color=svi_color, linestyle="--", linewidth=1.5, label="SVI"
            )

    from matplotlib.ticker import FixedFormatter, FixedLocator

    shown_lams = {0, 0.01, 0.05, 0.1, 0.5, 1.0}
    tick_positions = [i for i, lam in enumerate(all_lams) if lam in shown_lams]
    tick_labels = ["0" if all_lams[i] == 0 else str(all_lams[i]) for i in tick_positions]

    for ax in [ax_rmse, ax_but] if ax_sev is None else [ax_rmse, ax_but, ax_sev]:
        ax.set_xlim(-0.5, len(all_lams) - 0.5)
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax.set_xlabel(r"$\lambda$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_rmse.set_ylabel(r"RMSE$_\text{missing}$")
    ax_rmse.set_title("Reconstruction Error")

    ax_but.set_ylabel("Butterfly violation rate (%)")
    ax_but.set_title("Violation Rate")
    if gt:
        ax_but.axhline(
            gt["butterfly_rate"] * 100,
            color="#aaaaaa",
            linestyle="--",
            linewidth=1,
            label="GT",
        )

    if ax_sev is not None:
        ax_sev.set_ylabel(r"Expected severity ($\times 10^{-4}$)")
        ax_sev.set_title("Expected Severity")
        if gt:
            ax_sev.axhline(
                gt["butterfly_expected_severity"] * 1e4,
                color="#aaaaaa",
                linestyle="--",
                linewidth=1,
                label="GT",
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Recompute path (GPU required)
# ---------------------------------------------------------------------------


def recompute_regional_metrics() -> None:
    """Load checkpoints, run inference, compute per-region and per-surface metrics."""
    import torch

    from data.datasets import MaskConfig, VolSurfaceDataset
    from evaluation.comparison import (
        compute_regional_metrics,
        mean_absolute_error_grid,
        per_surface_rmse,
    )
    from evaluation.metrics import compute_metrics
    from experiments.train_baseline import build_model

    DATA_DIR = Path("data/synthetic/generated")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = VolSurfaceDataset(
        DATA_DIR / "test", mask_config=MaskConfig(mask_type="random", missing_frac=0.3)
    )
    log_m = test_ds.log_moneyness
    taus = test_ds.taus

    # Collect all targets/masks once
    all_targets, all_masks, all_tmsks = [], [], []
    all_inputs = []
    for i in range(len(test_ds)):
        inp, target, mask, target_mask = test_ds[i]
        all_inputs.append(inp)
        all_targets.append(target)
        all_masks.append(mask)
        all_tmsks.append(target_mask)

    target_stack = torch.stack(all_targets)
    mask_stack = torch.stack(all_masks)
    tmsk_stack = torch.stack(all_tmsks)

    models_to_eval = [
        ("U-Net", "unet", "synthetic"),
        ("Transformer", "transformer", "synthetic"),
        ("CNN", "cnn", "synthetic"),
        ("MLP", "mlp", "synthetic"),
    ]

    cache = {}

    for display, model_name, variant in models_to_eval:
        print(f"\n  Recomputing: {display}")
        ckpt = OUT_DIR / model_name / variant / "best_model.pt"
        if not ckpt.exists():
            print(f"    Checkpoint not found: {ckpt}, skipping")
            continue

        model = build_model(
            model_name,
            len(taus),
            len(test_ds.strikes),
            taus=taus,
            log_moneyness=log_m,
        )
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model = model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for inp in all_inputs:
                pred = model(inp.unsqueeze(0).to(device)).cpu()
                preds.append(pred.squeeze(0))
        pred_stack = torch.stack(preds)

        # Per-region
        regional = compute_regional_metrics(
            pred_stack, target_stack, mask_stack, log_m, taus, tmsk_stack
        )
        # Per-surface RMSE
        surf_rmse = per_surface_rmse(pred_stack, target_stack, mask_stack, tmsk_stack)
        # Error grid
        err_grid = mean_absolute_error_grid(pred_stack, target_stack, tmsk_stack)
        # Overall
        overall = compute_metrics(pred_stack, target_stack, mask_stack, tmsk_stack)

        cache[display] = {
            "regional": {
                dim: [
                    {
                        "region": r.region,
                        "rmse_missing": r.rmse_missing,
                        "rmse_all": r.rmse_all,
                        "mae": r.mae,
                        "n_points": r.n_points,
                    }
                    for r in regions
                ]
                for dim, regions in regional.items()
            },
            "per_surface_rmse": surf_rmse.tolist(),
            "error_grid": err_grid.tolist(),
            "overall_rmse_missing": overall.rmse_missing,
        }
        print(f"    RMSE missing: {overall.rmse_missing:.6f}")

    # Save cache
    cache_path = COMPARE_DIR / "regional_metrics.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\n  Cached to {cache_path}")

    return cache


def fig_error_heatmaps(cache: dict) -> matplotlib.figure.Figure | None:
    """Multi-panel error heatmap: mean |error| at each grid point."""
    models = [k for k in cache if "error_grid" in cache[k]]
    if not models:
        return None

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, constrained_layout=True)

    log_m = np.log(np.linspace(70, 130, 25) / 100.0)
    taus = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Shared colorbar range
    vmax = max(np.array(cache[m]["error_grid"]).max() for m in models)

    for i, name in enumerate(models):
        grid = np.array(cache[name]["error_grid"])
        ax = axes[0, i]
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            extent=[log_m[0], log_m[-1], taus[0], taus[-1]],
            vmin=0,
            vmax=vmax,
            cmap="YlOrRd",
        )
        ax.set_title(name)
        ax.set_xlabel("Log-moneyness")
        if i == 0:
            ax.set_ylabel(r"$\tau$")

    fig.colorbar(im, ax=axes[0, -1], label="Mean |error|", shrink=0.8)
    fig.suptitle("Spatial Error Distribution (Synthetic, 30% missing)")
    return fig


def fig_rmse_boxplots(cache: dict) -> matplotlib.figure.Figure | None:
    """Box plots of per-surface RMSE distribution."""
    models = [k for k in cache if "per_surface_rmse" in cache[k]]
    if not models:
        return None

    data = [cache[m]["per_surface_rmse"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data, labels=models, patch_artist=True, showfliers=True, flierprops={"markersize": 2}
    )

    for patch, name in zip(bp["boxes"], models, strict=False):
        color = MODEL_COLORS.get(name, "#333333")
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(r"RMSE$_\text{missing}$ (per surface)")
    ax.set_title("Per-Surface Reconstruction Error Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_regional_bar_chart(cache: dict) -> matplotlib.figure.Figure | None:
    """Grouped bar charts: RMSE by moneyness and tenor region."""
    models = [k for k in cache if "regional" in cache[k]]
    if not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Moneyness
    money_regions = [r["region"] for r in cache[models[0]]["regional"]["moneyness"]]
    x = np.arange(len(money_regions))
    width = 0.8 / len(models)

    for i, name in enumerate(models):
        vals = [r["rmse_missing"] for r in cache[name]["regional"]["moneyness"]]
        color = MODEL_COLORS.get(name, "#333333")
        ax1.bar(x + i * width, vals, width, label=name, color=color)

    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels([r.replace("_", " ") for r in money_regions], rotation=30, ha="right")
    ax1.set_ylabel(r"RMSE$_\text{missing}$")
    ax1.set_title("By Moneyness Region")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Tenor
    tenor_regions = [r["region"] for r in cache[models[0]]["regional"]["tenor"]]
    x = np.arange(len(tenor_regions))

    for i, name in enumerate(models):
        vals = [r["rmse_missing"] for r in cache[name]["regional"]["tenor"]]
        color = MODEL_COLORS.get(name, "#333333")
        ax2.bar(x + i * width, vals, width, label=name, color=color)

    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(tenor_regions)
    ax2.set_ylabel(r"RMSE$_\text{missing}$")
    ax2.set_title("By Tenor Region")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Regional Error Analysis (Synthetic, 30% missing)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Qualitative figures (recompute path — GPU required)
# ---------------------------------------------------------------------------


def recompute_qualitative_figures() -> None:
    """Generate sample reconstruction, attention, and smile-slice figures.

    Loads model checkpoints, selects a test surface with large SVI-Transformer
    gap (filtered for normal-looking surfaces), runs inference for all models,
    and produces three publication-quality PDF figures.
    """
    import torch

    from data.datasets import MaskConfig, VolSurfaceDataset
    from evaluation.comparison import per_surface_rmse
    from experiments.train_baseline import build_model
    from models.svi.calibration import calibrate_surface
    from models.svi.svi import svi_iv

    DATA_DIR = Path("data/synthetic/generated")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = VolSurfaceDataset(
        DATA_DIR / "test",
        mask_config=MaskConfig(mask_type="random", missing_frac=0.3),
        seed=42,
    )
    log_m = test_ds.log_moneyness
    taus = test_ds.taus

    # --- Load Transformer and find median-RMSE surface ---
    ckpt_path = OUT_DIR / "transformer" / "synthetic" / "best_model.pt"
    if not ckpt_path.exists():
        print("  Transformer checkpoint not found, skipping qualitative figures")
        return

    trans_model = build_model(
        "transformer", len(taus), len(test_ds.strikes), taus=taus, log_moneyness=log_m
    )
    trans_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    trans_model = trans_model.to(device).eval()

    # Collect all test data
    all_inputs, all_targets, all_masks, all_tmsks = [], [], [], []
    for i in range(len(test_ds)):
        inp, target, mask, tmsk = test_ds[i]
        all_inputs.append(inp)
        all_targets.append(target)
        all_masks.append(mask)
        all_tmsks.append(tmsk)

    target_stack = torch.stack(all_targets)
    mask_stack = torch.stack(all_masks)
    tmsk_stack = torch.stack(all_tmsks)

    # Run Transformer inference on full test set
    trans_preds = []
    with torch.no_grad():
        for inp in all_inputs:
            pred = trans_model(inp.unsqueeze(0).to(device)).cpu()
            trans_preds.append(pred.squeeze(0))
    trans_pred_stack = torch.stack(trans_preds)

    # --- Load all neural models and run inference on full test set ---
    nn_models = {"Transformer": (trans_preds, trans_pred_stack)}
    for model_name, dir_name in [("U-Net", "unet"), ("CNN", "cnn")]:
        ckpt = OUT_DIR / dir_name / "synthetic" / "best_model.pt"
        if not ckpt.exists():
            print(f"  {model_name} checkpoint not found, skipping")
            continue
        m = build_model(dir_name, len(taus), len(test_ds.strikes), taus=taus, log_moneyness=log_m)
        m.load_state_dict(torch.load(ckpt, weights_only=True))
        m = m.to(device).eval()
        preds = []
        with torch.no_grad():
            for inp in all_inputs:
                pred = m(inp.unsqueeze(0).to(device)).cpu()
                preds.append(pred.squeeze(0))
        nn_models[model_name] = (preds, torch.stack(preds))

    # Per-surface RMSE for each model
    nn_rmses = {}
    for name, (_, pred_stack) in nn_models.items():
        nn_rmses[name] = per_surface_rmse(pred_stack, target_stack, mask_stack, tmsk_stack)

    # Select surface where spread between neural models is largest,
    # filtered for normal-looking surfaces and no extreme model artifacts
    model_rmse_matrix = np.stack(list(nn_rmses.values()))  # (n_models, n_surfaces)
    spread = model_rmse_matrix.max(axis=0) - model_rmse_matrix.min(axis=0)
    valid = []
    for i in range(len(test_ds)):
        gt_i = all_targets[i][0].numpy()
        if gt_i.min() <= 0.10 or (gt_i.max() - gt_i.min()) <= 0.02:
            continue
        # Reject surfaces where any model produces extreme artifacts
        ok = True
        for _, (preds, _) in nn_models.items():
            pred_i = preds[i][0].numpy()
            if pred_i.min() < gt_i.min() - 0.03 or pred_i.max() > gt_i.max() + 0.03:
                ok = False
                break
        if ok:
            valid.append(i)
    valid = np.array(valid, dtype=int)
    sel_idx = int(valid[np.argmax(spread[valid])])

    per_model_str = ", ".join(f"{n}={nn_rmses[n][sel_idx]:.4f}" for n in nn_rmses)
    print(
        f"  Selected surface: idx={sel_idx}, spread={spread[sel_idx]:.6f}, "
        f"{per_model_str}, "
        f"IV range=[{all_targets[sel_idx][0].min():.3f}, "
        f"{all_targets[sel_idx][0].max():.3f}]"
    )

    # Extract data for selected surface
    inp_tensor = all_inputs[sel_idx]
    gt = all_targets[sel_idx][0].numpy()
    obs_mask = inp_tensor[1].numpy().astype(bool)

    model_preds = {}
    for name, (preds, _) in nn_models.items():
        model_preds[name] = preds[sel_idx][0].numpy()

    # SVI for selected surface
    masked_iv = inp_tensor[0].numpy()
    params_list = calibrate_surface(log_m, masked_iv, taus, obs_mask)
    svi_pred = np.zeros_like(gt)
    for j, (params, tau) in enumerate(zip(params_list, taus, strict=True)):
        svi_pred[j] = svi_iv(log_m, float(tau), params)
    model_preds["SVI"] = svi_pred

    # --- Generate figures ---
    fig = fig_sample_reconstruction(gt, obs_mask, model_preds, taus, log_m)
    save_fig(fig, "sample_reconstruction")

    fig = fig_smile_slices(gt, model_preds, obs_mask, taus, log_m)
    save_fig(fig, "smile_slices")

    # Attention heatmap (Transformer only)
    fig = fig_attention_heatmap(trans_model, inp_tensor, obs_mask, taus, log_m, device)
    save_fig(fig, "attention_heatmap")


def fig_sample_reconstruction(
    gt: np.ndarray,
    obs_mask: np.ndarray,
    model_preds: dict[str, np.ndarray],
    taus: np.ndarray,
    log_m: np.ndarray,
) -> matplotlib.figure.Figure:
    """2x3 heatmap: GT, Masked Input, Transformer / U-Net, CNN, SVI."""
    vmin, vmax = gt.min(), gt.max()

    # Build masked input for display
    masked_display = np.where(obs_mask, gt, np.nan)

    panels = [
        ("Ground Truth", gt),
        ("Masked Input", masked_display),
        ("Transformer", model_preds.get("Transformer")),
        ("U-Net", model_preds.get("U-Net")),
        ("CNN", model_preds.get("CNN")),
        ("SVI", model_preds.get("SVI")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    for ax, (title, data) in zip(axes.flat, panels, strict=True):
        if data is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Log-moneyness")
        ax.set_ylabel(r"$\tau$ (years)")
        # X-axis: log-moneyness ticks
        n_lm = len(log_m)
        lm_ticks = np.linspace(0, n_lm - 1, 5)
        lm_labels = [f"{log_m[int(round(t))]:.1f}" for t in lm_ticks]
        ax.set_xticks(lm_ticks)
        ax.set_xticklabels(lm_labels)
        # Y-axis: actual tenor values (uniform cell height)
        ax.set_yticks(range(len(taus)))
        ax.set_yticklabels([f"{t:.2f}" for t in taus])

    fig.colorbar(im, ax=axes, label="Implied Volatility", shrink=0.8)
    return fig


def fig_attention_heatmap(
    model,
    inp_tensor,
    obs_mask: np.ndarray,
    taus: np.ndarray,
    log_m: np.ndarray,
    device=None,
) -> matplotlib.figure.Figure | None:
    """1x3 attention heatmap for 3 representative missing tokens."""
    import torch

    from experiments.attention_utils import capture_cross_attention

    if device is None:
        device = next(model.parameters()).device

    n_taus, n_strikes = len(taus), len(log_m)
    flat_mask = obs_mask.flatten()  # (200,)
    missing_indices = np.where(~flat_mask)[0]

    if len(missing_indices) < 3:
        print("  Too few missing tokens for attention figure")
        return None

    # Select 3 representative missing tokens by grid position
    # Target: ATM short, OTM long, deep OTM wing
    target_positions = [
        (2, n_strikes // 2),  # tau=0.25, ATM
        (7, 4),  # tau=2.0, OTM put
        (3, 20),  # tau=0.5, OTM call
    ]
    query_indices = []
    query_labels = []
    for ti, ki in target_positions:
        flat_idx = ti * n_strikes + ki
        if not flat_mask[flat_idx]:  # token is actually missing
            query_indices.append(flat_idx)
            query_labels.append(f"$\\tau$={taus[ti]:.2f}, $k$={log_m[ki]:.2f}")
        else:
            # Find nearest missing token
            dists = np.abs(missing_indices - flat_idx)
            nearest = missing_indices[np.argmin(dists)]
            query_indices.append(nearest)
            ni_tau = nearest // n_strikes
            ni_k = nearest % n_strikes
            query_labels.append(f"$\\tau$={taus[ni_tau]:.2f}, $k$={log_m[ni_k]:.2f}")

    # Run inference with attention capture
    model.eval()
    with torch.no_grad():
        with capture_cross_attention(model) as attn_weights:
            model(inp_tensor.unsqueeze(0).to(device))

    # Use last decoder layer, head-averaged
    last_layer = max(attn_weights.keys())
    attn = attn_weights[last_layer][0].cpu().numpy()  # (tgt_len, src_len)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, qi, label in zip(axes, query_indices, query_labels, strict=True):
        # Reshape attention weights to grid
        attn_grid = attn[qi].reshape(n_taus, n_strikes)

        im = ax.imshow(
            attn_grid,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            extent=[log_m[0], log_m[-1], taus[0], taus[-1]],
        )
        ax.set_title(f"Query: {label}", fontsize=9)
        ax.set_xlabel("Log-moneyness")
        ax.set_ylabel(r"$\tau$ (years)")

        # Mark the query token position
        qi_tau = taus[qi // n_strikes]
        qi_k = log_m[qi % n_strikes]
        ax.plot(qi_k, qi_tau, "s", color="blue", markersize=8, markeredgecolor="white", zorder=5)

        # Mark missing positions with small dots
        for mi in missing_indices:
            mt = taus[mi // n_strikes]
            mk = log_m[mi % n_strikes]
            ax.plot(mk, mt, ".", color="gray", markersize=2, alpha=0.3)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Transformer Cross-Attention Weights (Last Decoder Layer)", fontsize=11)
    fig.subplots_adjust(top=0.88, wspace=0.4)
    return fig


def fig_smile_slices(
    gt: np.ndarray,
    model_preds: dict[str, np.ndarray],
    obs_mask: np.ndarray,
    taus: np.ndarray,
    log_m: np.ndarray,
    target_taus: list[float] | None = None,
) -> matplotlib.figure.Figure:
    """1x3 panel: GT vs model predictions at 3 tenors."""
    if target_taus is None:
        target_taus = [0.08, 0.75, 2.0]
    tau_indices = [int(np.argmin(np.abs(taus - t))) for t in target_taus]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, ti in zip(axes, tau_indices, strict=True):
        tau_val = taus[ti]
        slice_mask = obs_mask[ti].astype(bool)

        # Ground truth
        ax.plot(log_m, gt[ti], "k-", linewidth=2, label="Ground Truth")

        # Mark observed vs missing
        ax.plot(
            log_m[slice_mask], gt[ti][slice_mask], "ko", markersize=4, label="Observed", zorder=5
        )
        ax.plot(
            log_m[~slice_mask],
            gt[ti][~slice_mask],
            "kx",
            markersize=4,
            label="Missing",
            zorder=5,
        )

        # Model predictions
        for name in ["Transformer", "U-Net", "CNN", "SVI"]:
            pred = model_preds.get(name)
            if pred is not None:
                color = MODEL_COLORS.get(name, "gray")
                ax.plot(log_m, pred[ti], "--", color=color, linewidth=1.5, label=name)

        ax.set_title(f"$\\tau$ = {tau_val:.2f}y")
        ax.set_xlabel("Log-moneyness")
        ax.set_ylabel("Implied Volatility")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig


def generate_computational_table() -> str | None:
    """Read benchmark.json and format as booktabs LaTeX table."""
    bench_path = COMPARE_DIR / "benchmark.json"
    if not bench_path.exists():
        return None
    with open(bench_path) as f:
        data = json.load(f)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Params & Latency (ms) & Throughput (surf/s) & GPU Memory (MB) \\",
        r"\midrule",
    ]
    for name, info in data.items():
        params = info.get("n_params", "--")
        if isinstance(params, int):
            params = f"{params // 1000}k"
        lat = info.get("latency_ms", {})
        lat_str = f"{lat.get('mean', 0):.1f}" if lat else "--"
        tp = info.get("throughput_per_sec", 0)
        tp_str = f"{tp:.0f}" if tp else "--"
        mem = info.get("gpu_memory_mb", 0)
        mem_str = f"{mem:.1f}" if mem else "--"
        lines.append(f"{name} & {params} & {lat_str} & {tp_str} & {mem_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def save_fig(fig: matplotlib.figure.Figure | None, name: str) -> None:
    """Save figure as PDF."""
    if fig is None:
        return
    path = COMPARE_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis comparison outputs")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run inference for per-region metrics (needs GPU)",
    )
    args = parser.parse_args()

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(THESIS_RC)

    do_tables = not args.figures_only
    do_figures = not args.tables_only

    # --- Tables ---
    if do_tables:
        print("Generating LaTeX tables...")

        tex = generate_synthetic_table(SYNTHETIC_MODELS)
        (COMPARE_DIR / "table_synthetic.tex").write_text(tex)
        print(f"  Saved {COMPARE_DIR / 'table_synthetic.tex'}")

        tex = generate_real_table()
        (COMPARE_DIR / "table_real.tex").write_text(tex)
        print(f"  Saved {COMPARE_DIR / 'table_real.tex'}")

        tex = generate_arbitrage_table()
        (COMPARE_DIR / "table_arbitrage.tex").write_text(tex)
        print(f"  Saved {COMPARE_DIR / 'table_arbitrage.tex'}")

    # --- Figures from pre-computed metrics ---
    if do_figures:
        print("\nGenerating figures from pre-computed metrics...")
        save_fig(fig_rmse_bar_chart(), "rmse_bar_chart")
        save_fig(fig_rmse_bar_chart_real(), "rmse_bar_chart_real")
        save_fig(fig_pareto_accuracy_vs_arbitrage(), "pareto_accuracy_arbitrage")
        save_fig(fig_masking_degradation(), "masking_degradation")
        save_fig(fig_transfer_waterfall(), "transfer_waterfall")
        save_fig(fig_constraint_impact(), "constraint_impact")
        save_fig(fig_pareto_lambda_sweep(), "pareto_lambda_sweep")

    # --- Computational table (from benchmark.json, no GPU needed) ---
    if do_tables:
        tex = generate_computational_table()
        if tex:
            (COMPARE_DIR / "table_computational.tex").write_text(tex)
            print(f"  Saved {COMPARE_DIR / 'table_computational.tex'}")

    # --- Recompute path (GPU) ---
    if args.recompute:
        print("\nRecomputing per-region metrics (GPU)...")
        cache = recompute_regional_metrics()

        if do_figures and cache:
            print("\nGenerating recompute figures...")
            save_fig(fig_error_heatmaps(cache), "error_heatmaps")
            save_fig(fig_rmse_boxplots(cache), "rmse_boxplots")
            save_fig(fig_regional_bar_chart(cache), "regional_bar_chart")

        print("\nGenerating qualitative figures (GPU)...")
        recompute_qualitative_figures()

        # Regional table
        if do_tables and cache:
            # Generate regional table from cache
            models = list(cache.keys())
            if models:
                lines = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule"]
                header = "Region & " + " & ".join(models) + r" \\"
                lines.append(header)
                lines.append(r"\midrule")

                # Moneyness
                for j, region in enumerate(cache[models[0]]["regional"]["moneyness"]):
                    row = region["region"].replace("_", " ")
                    for m in models:
                        val = cache[m]["regional"]["moneyness"][j]["rmse_missing"]
                        row += f" & {val:.4f}"
                    lines.append(row + r" \\")

                lines.append(r"\midrule")

                # Tenor
                for j, region in enumerate(cache[models[0]]["regional"]["tenor"]):
                    row = region["region"]
                    for m in models:
                        val = cache[m]["regional"]["tenor"][j]["rmse_missing"]
                        row += f" & {val:.4f}"
                    lines.append(row + r" \\")

                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
                tex = "\n".join(lines)
                (COMPARE_DIR / "table_regional.tex").write_text(tex)
                print(f"  Saved {COMPARE_DIR / 'table_regional.tex'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
