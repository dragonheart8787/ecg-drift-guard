"""All matplotlib visualisations (no seaborn dependency).

Covers: reliability, drift curves, confidence dist, reject curves,
intensity curves, correlation scatter, top-feature shift, class dist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "blue": "#4c72b0",
    "red": "#c44e52",
    "green": "#55a868",
    "orange": "#dd8452",
    "purple": "#8172b3",
    "grey": "#937860",
}


def _savefig(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  1. RELIABILITY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════

def plot_reliability(rel_data: dict, ece: float, title: str, path: str | Path) -> None:
    accs = rel_data["bin_accs"]
    n_bins = len(accs)
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(5, 5))
    xs = np.arange(n_bins) * width + width / 2
    ax.bar(xs, accs, width=width * 0.9, color=COLORS["blue"], edgecolor="white", label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}  (ECE={ece:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    _savefig(fig, path)


def plot_reliability_comparison(
    rel_before: dict, ece_before: float,
    rel_after: dict, ece_after: float,
    path: str | Path,
) -> None:
    """Side-by-side reliability diagrams (before vs after calibration)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rel, ece_val, title in [
        (ax1, rel_before, ece_before, "Before Calibration"),
        (ax2, rel_after, ece_after, "After Calibration"),
    ]:
        n_bins = len(rel["bin_accs"])
        w = 1.0 / n_bins
        xs = np.arange(n_bins) * w + w / 2
        ax.bar(xs, rel["bin_accs"], width=w * 0.9, color=COLORS["blue"], edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{title}  (ECE={ece_val:.4f})")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  2. DRIFT CURVE (per scenario, single intensity)
# ══════════════════════════════════════════════════════════════════════════

def plot_drift_curve(drift_results: list[dict], path: str | Path) -> None:
    labels = [f"{d['scenario']}@{d.get('intensity','')}" for d in drift_results]
    scores = [d["drift_score"] for d in drift_results]
    drops = [d["perf_drop_f1"] for d in drift_results]

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax1.bar(x - 0.15, scores, 0.3, color=COLORS["blue"], label="Drift score")
    ax1.set_ylabel("Drift score", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])

    ax2 = ax1.twinx()
    ax2.bar(x + 0.15, drops, 0.3, color=COLORS["red"], label="F1 drop")
    ax2.set_ylabel("F1 macro drop", color=COLORS["red"])
    ax2.tick_params(axis="y", labelcolor=COLORS["red"])

    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Drift Score vs Performance Drop")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  3. DRIFT INTENSITY CURVE (S1→S2→S3 for each scenario)
# ══════════════════════════════════════════════════════════════════════════

def plot_intensity_curve(intensity_curve: dict, path: str | Path) -> None:
    """Line plot: intensity (x) vs drift_score & f1_drop for each scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    markers = ["o", "s", "^", "D", "v"]

    for i, (scenario, curve) in enumerate(intensity_curve.items()):
        ints = curve["intensities"]
        ax1.plot(ints, curve["drift_scores"], marker=markers[i % len(markers)],
                 label=f"{scenario} (emb)")
        ax1.plot(ints, curve["baseline_drift_scores"], marker=markers[i % len(markers)],
                 linestyle="--", alpha=0.5, label=f"{scenario} (baseline B)")
        ax2.plot(ints, curve["f1_drops"], marker=markers[i % len(markers)],
                 label=scenario)

    ax1.set_xlabel("Intensity"); ax1.set_ylabel("Drift score")
    ax1.set_title("Drift Score by Intensity (Embedding vs Baseline B)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Intensity"); ax2.set_ylabel("F1 drop")
    ax2.set_title("Performance Drop by Intensity")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  4. CORRELATION SCATTER (drift_score vs F1 drop) — H1 evidence
# ══════════════════════════════════════════════════════════════════════════

def plot_correlation_scatter(
    drift_results: list[dict],
    corr_info: dict,
    path: str | Path,
) -> None:
    """Scatter with Spearman ρ annotation, colored by scenario."""
    fig, ax = plt.subplots(figsize=(6, 5))
    scenario_colors = {}
    color_list = list(COLORS.values())

    for d in drift_results:
        sc = d["scenario"]
        if sc not in scenario_colors:
            scenario_colors[sc] = color_list[len(scenario_colors) % len(color_list)]
        c = scenario_colors[sc]
        ax.scatter(d["drift_score"], d["perf_drop_f1"], color=c, s=80, zorder=3,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(f"{sc}@{d.get('intensity','')}",
                    (d["drift_score"], d["perf_drop_f1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    sr = corr_info.get("spearman_r", "N/A")
    sp = corr_info.get("spearman_p", "N/A")
    ax.set_xlabel("Drift score (embedding)")
    ax.set_ylabel("F1 macro drop")
    ax.set_title(f"H1: Drift Score vs Performance Drop\nSpearman ρ={sr}, p={sp}")
    ax.grid(True, alpha=0.3)

    if isinstance(sr, (int, float)) and sr is not None:
        xs = np.array([d["drift_score"] for d in drift_results])
        if len(xs) > 1:
            m, b = np.polyfit(xs, [d["perf_drop_f1"] for d in drift_results], 1)
            xx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xx, m * xx + b, "k--", alpha=0.4, lw=1)

    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  5. PERFORMANCE VS DRIFT (original simple version)
# ══════════════════════════════════════════════════════════════════════════

def plot_perf_vs_drift(drift_results: list[dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for d in drift_results:
        ax.scatter(d["drift_score"], d.get("f1_drift", 0), s=80, zorder=3)
        label = f"{d['scenario']}@{d.get('intensity','')}"
        ax.annotate(label, (d["drift_score"], d.get("f1_drift", 0)),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Drift score"); ax.set_ylabel("F1 macro (drifted)")
    ax.set_title("Model Performance vs Drift Score")
    ax.grid(True, alpha=0.3)
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  6. TOP FEATURE SHIFT
# ══════════════════════════════════════════════════════════════════════════

def plot_top_feature_shift(per_dim_psi: list[float], top_dims: list[int], path: str | Path) -> None:
    dims = top_dims[:10]
    vals = [per_dim_psi[d] for d in dims]
    labels = [f"dim {d}" for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(dims))
    ax.barh(y_pos, vals, color=COLORS["green"], edgecolor="white")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_xlabel("PSI")
    ax.set_title("Top Embedding Dimensions by PSI Shift")
    ax.invert_yaxis()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  7. CONFIDENCE DISTRIBUTION (correct vs incorrect)
# ══════════════════════════════════════════════════════════════════════════

def plot_confidence_distribution(conf_analysis: dict, path: str | Path) -> None:
    """Overlapping histograms of confidence for correct vs incorrect predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in [
        (axes[0], "conf", "Confidence"),
        (axes[1], "entropy", "Entropy"),
        (axes[2], "margin", "Margin"),
    ]:
        correct = conf_analysis.get(f"{key}_correct", [])
        incorrect = conf_analysis.get(f"{key}_incorrect", [])
        if correct:
            ax.hist(correct, bins=40, alpha=0.6, color=COLORS["blue"],
                    label="Correct", density=True)
        if incorrect:
            ax.hist(incorrect, bins=40, alpha=0.6, color=COLORS["red"],
                    label="Incorrect", density=True)
        ax.set_title(f"{title} Distribution")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  8. REJECT CURVE (reject rate vs error rate) — H3 evidence
# ══════════════════════════════════════════════════════════════════════════

def plot_reject_curve(reject_data: dict, path: str | Path) -> None:
    """As reject_rate ↑, error_rate should ↓."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(reject_data["reject_rates"], reject_data["error_rates"],
            color=COLORS["blue"], lw=2)
    ax.set_xlabel("Reject rate")
    ax.set_ylabel("Error rate (among accepted)")
    ax.set_title("H3: Reject Rate vs Error Rate\n(lower-right = better trade-off)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, max(reject_data["error_rates"]) * 1.1 + 0.01)
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
#  9. SELECTIVE PREDICTION CURVE (coverage vs risk)
# ══════════════════════════════════════════════════════════════════════════

def plot_coverage_risk(sel_data: dict, path: str | Path) -> None:
    """Coverage (x) vs Risk / Accuracy (y)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sel_data["coverages"], sel_data["risks"],
            color=COLORS["red"], lw=2, label="Risk (error rate)")
    ax.plot(sel_data["coverages"], sel_data["accuracies"],
            color=COLORS["green"], lw=2, label="Accuracy")
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Rate")
    ax.set_title("Selective Prediction: Coverage vs Risk/Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
# 10. CLASS DISTRIBUTION BAR CHART
# ══════════════════════════════════════════════════════════════════════════

def plot_class_distribution(class_dist: dict, path: str | Path) -> None:
    """Grouped bar chart of class distribution across splits."""
    splits = [s for s in class_dist if not s.startswith("_")]
    if not splits:
        return
    sample_split = class_dist[splits[0]]
    class_names = [k for k in sample_split if k != "_total"]

    n_splits = len(splits)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / n_splits
    color_list = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, split in enumerate(splits):
        ratios = [class_dist[split][c]["ratio"] for c in class_names]
        ax.bar(x + i * width, ratios, width, label=split,
               color=color_list[i % len(color_list)], edgecolor="white")

    ax.set_xticks(x + width * (n_splits - 1) / 2)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Proportion")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
# 11. CALIBRATION METRICS TABLE (as figure)
# ══════════════════════════════════════════════════════════════════════════

def plot_calibration_table(cal_info: dict, path: str | Path) -> None:
    """Render ECE + Brier before/after as a table image."""
    rows = [
        ["ECE", f"{cal_info.get('ece_before', '-'):.4f}", f"{cal_info.get('ece_after', '-'):.4f}"],
        ["Brier", f"{cal_info.get('brier_before', '-'):.4f}", f"{cal_info.get('brier_after', '-'):.4f}"],
    ]
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Before", "After"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax.set_title("Calibration Metrics (Before vs After Temperature Scaling)", pad=20)
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
# 12. OOD SCORE DISTRIBUTION (internal vs external)
# ══════════════════════════════════════════════════════════════════════════

def plot_ood_distribution(
    ref_scores: np.ndarray,
    ext_scores: np.ndarray,
    threshold: float,
    path: str | Path,
) -> None:
    """Overlapping histograms of Mahalanobis OOD scores."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ref_scores, bins=50, alpha=0.6, color=COLORS["blue"],
            label="In-distribution (MIT-BIH test)", density=True)
    ax.hist(ext_scores, bins=50, alpha=0.6, color=COLORS["red"],
            label="External dataset", density=True)
    ax.axvline(threshold, color="k", linestyle="--", lw=1.5,
               label=f"OOD threshold ({threshold:.2f})")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Density")
    ax.set_title("OOD Score Distribution: Internal vs External")
    ax.legend()
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
# 13. EXTERNAL DRIFT COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def plot_external_drift_comparison(
    internal_drift_results: list[dict],
    external_drift: dict,
    ext_db_name: str,
    path: str | Path,
) -> None:
    """Bar chart comparing drift scores: simulated scenarios vs real external."""
    labels = [f"{d['scenario']}@{d.get('intensity','')}" for d in internal_drift_results]
    scores = [d["drift_score"] for d in internal_drift_results]

    labels.append(f"EXTERNAL\n({ext_db_name})")
    scores.append(external_drift["score"])

    x = np.arange(len(labels))
    colors = [COLORS["blue"]] * (len(labels) - 1) + [COLORS["red"]]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5))
    ax.bar(x, scores, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Drift score")
    ax.set_title("Drift Score: Simulated Scenarios vs Real External Dataset")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, path)


# ══════════════════════════════════════════════════════════════════════════
# 14. LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════

def plot_latency(
    model_lat: list[dict],
    emb_lat: list[dict],
    path: str | Path,
) -> None:
    """Batch size vs ms/beat for model and embedder."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bs_m = [d["batch_size"] for d in model_lat]
    ms_m = [d["per_beat_ms"] for d in model_lat]
    bs_e = [d["batch_size"] for d in emb_lat]
    ms_e = [d["per_beat_ms"] for d in emb_lat]

    ax.plot(bs_m, ms_m, "o-", color=COLORS["blue"], lw=2, label="CNN (full)")
    ax.plot(bs_e, ms_e, "s--", color=COLORS["green"], lw=2, label="Embedder")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms / beat)")
    ax.set_title("Inference Latency by Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    _savefig(fig, path)
