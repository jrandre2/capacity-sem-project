"""
Generate two main-text figures requested by R7:

1. Specification-curve / coefficient-spectrum figure (R7 minor #5):
   Plot the 12-row robustness summary as Burden->Timeliness beta with
   95% CI, color-coded by Class (I, II, III).

2. QCEW zero-rate / denominator-behavior figure (R7 major #3):
   Histogram showing zero-share by state vs. local + the staffing-scaled
   denominator distribution, illustrating that for ~80% of the sample
   the denominator collapses to the epsilon-offset value.

Outputs:
  manuscript_quarto/figures/fig_02_specification_curve.png
  manuscript_quarto/figures/fig_03_qcew_denominator.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "manuscript_quarto" / "figures"
WORKTREE_FIG_DIR = (
    PROJECT_ROOT
    / ".claude"
    / "worktrees"
    / "festive-cannon"
    / "manuscript_quarto"
    / "figures"
)
SEM_DATA_PATH = (
    PROJECT_ROOT
    / "data_work"
    / "diagnostics"
    / "kaifa_recovered_analysis"
    / "recovered_notebook_reference_573_sem_ready_dataset.csv"
)

warnings.filterwarnings("ignore")


def make_specification_curve():
    """Coefficient-spectrum plot for the robustness dashboard rows."""

    # Hardcode the table from index.qmd to keep this script self-contained.
    # If any value changes in the manuscript, update both places.
    rows = [
        # (label, class, n, beta, ci_lo, ci_hi, note)
        ("Reference two-factor", "Reference", 573, 0.266, 0.151, 0.352, "ML SE"),
        ("Reference + state-clustered bootstrap (n=1000)", "Ia", 573, 0.266, -0.129, 0.531, "Cluster SE"),
        ("QCEW bounds: zeros imputed 0.25–10 emp./K", "Ia", 573, 0.146, 0.024, 0.267, "range"),
        ("Drop weak indicator", "Ia", 573, 0.266, None, None, ""),
        ("Residualized burden", "Ia", 573, 0.297, None, None, ""),
        ("Maturity-band controls", "Ia", 573, 0.105, None, None, ""),
        ("Portfolio-scale controls", "Ia", 573, 0.127, None, None, ""),
        ("Local jurisdictions only (PRIMARY)", "Ib", 543, 0.257, None, None, "primary"),
        ("Nonzero-QCEW subsample", "Ib", 111, 0.132, None, None, "n.s."),
        ("Local-only AND nonzero-QCEW", "Ib", 81, 0.145, None, None, "n.s."),
        ("Mature portfolios only (>48 mo.)", "Ib", 275, -0.244, None, None, ""),
        ("Drop NDR/MIT-exposed states (>25%)", "Ib", 512, 0.255, None, None, ""),
        ("Raw portfolio counts (workload volume)", "II-C", 573, -0.443, None, None, ""),
        ("Financial-ratio capacity bridge (II-C*)", "II-C", 573, -0.600, None, None, ""),
        ("Reconstructed-panel fixed-horizon q=8", "II-O", 128, 0.003, None, None, "n.s."),
        ("Reconstructed-panel fixed-horizon q=12", "II-O", 116, -0.001, None, None, "n.s."),
        ("Reconstructed-panel fixed-horizon q=16", "II-O", 102, 0.025, None, None, "n.s."),
    ]

    df = pd.DataFrame(rows, columns=["label", "class", "n", "beta", "ci_lo", "ci_hi", "note"])
    df = df.iloc[::-1].reset_index(drop=True)  # plot top-down

    color_map = {
        "Reference": "#1f77b4",
        "Ia": "#2ca02c",
        "Ib": "#66c266",
        "II-C": "#d62728",
        "II-O": "#ff7f0e",
        "III": "#9467bd",
    }
    colors = df["class"].map(color_map)

    fig, ax = plt.subplots(figsize=(9.0, 8.5))
    y = np.arange(len(df))

    # Reference vertical line at zero
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, zorder=1)
    # Reference vertical line at point estimate
    ax.axvline(0.266, color="#1f77b4", linestyle=":", linewidth=0.8, zorder=1, label="Reference β = 0.266")
    # 50% magnitude band
    ax.axvspan(0.133, 0.266 * 1.5, color="#1f77b4", alpha=0.05, zorder=0)

    # Point estimates with optional CIs
    for i, row in df.iterrows():
        if row["ci_lo"] is not None and row["ci_hi"] is not None:
            ax.plot(
                [row["ci_lo"], row["ci_hi"]], [i, i],
                color=colors.iloc[i], linewidth=2.0, alpha=0.7, zorder=2,
            )
        ax.scatter(row["beta"], i, s=70, color=colors.iloc[i], zorder=3, edgecolor="black", linewidth=0.5)

    # Labels and class column
    labels = [f"{row['label']}  (N={row['n']})" for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Class column on right side
    ax2 = ax.twinx()
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["class"], fontsize=8)
    ax2.set_ylabel("Class", fontsize=9)
    ax2.set_ylim(ax.get_ylim())

    ax.set_xlabel("Standardized β: AdminBurdenCapacity → RecoveryTimeliness", fontsize=10)
    ax.set_xlim(-0.75, 0.65)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend for class colors
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["Reference"],
                   markersize=8, label="Reference"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["Ia"],
                   markersize=8, label="Class Ia (measurement-preserving)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["Ib"],
                   markersize=8, label="Class Ib (sample-scope)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["II-C"],
                   markersize=8, label="Class II-C (capacity-op change)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["II-O"],
                   markersize=8, label="Class II-O (outcome change)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)

    plt.tight_layout()
    out_paths = [
        FIG_DIR / "fig_02_specification_curve.png",
        WORKTREE_FIG_DIR / "fig_02_specification_curve.png",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def make_qcew_denominator_figure():
    """Histogram showing QCEW zero-rate by gov type and denominator distribution."""

    sem = pd.read_csv(SEM_DATA_PATH)
    # Recover unstandardized avg_employment proxy via inverse-z requires raw data;
    # easier: use z_avg_employment to count zero-band (within ~one SD of the
    # all-zero mode for locals). The actual pre-z value is approximated.

    # For panel A, compute share at zero by gov type from the parquet/raw if
    # available, else fall back to the documented fractions.
    # Attempt to load raw bundled dataset (which has avg_employment unstandardized).
    raw_csv = (
        PROJECT_ROOT / "manuscript_kaifa_archive" / "data" / "recovered_code_bundle"
        / "extracted" / "all_state_local_fund_latent_var_4_v2.csv"
    )
    raw = pd.read_csv(raw_csv)
    # Match SEM jurisdictions only
    raw = raw[raw["Grantee"].isin(sem["Grantee"])].copy()
    raw["state_level"] = raw["Grantee"].str.fullmatch(r"[A-Z]{2}").astype(int)

    # Panel A: zero-share bar chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    state_emp = raw[raw["state_level"] == 1]["avg_employment"]
    local_emp = raw[raw["state_level"] == 0]["avg_employment"]
    state_zero = (state_emp == 0).mean()
    local_zero = (local_emp == 0).mean()
    overall_zero = (raw["avg_employment"] == 0).mean()

    bars = ax.bar(
        ["State agencies\n(N=30)", "Local jurisdictions\n(N=543)", "Pooled\n(N=573)"],
        [state_zero * 100, local_zero * 100, overall_zero * 100],
        color=["#1f77b4", "#ff7f0e", "#7f7f7f"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_ylabel("% with zero QCEW employment\n(NAICS 925110)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("(a) Proxy zero-rate by government type", fontsize=10)
    for bar, val in zip(bars, [state_zero, local_zero, overall_zero]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=10,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: denominator distribution
    ax = axes[1]
    eps = 1e-6
    raw["denominator"] = raw["avg_employment"] * raw["E_TOTPOP"] + eps

    # Use log scale; explicitly mark the epsilon floor
    den = raw["denominator"]
    log_den = np.log10(den.replace(0, eps))
    ax.hist(
        [
            log_den[raw["state_level"] == 1],
            log_den[raw["state_level"] == 0],
        ],
        bins=40,
        stacked=True,
        color=["#1f77b4", "#ff7f0e"],
        label=["State agencies", "Local jurisdictions"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(np.log10(eps), color="red", linestyle="--", linewidth=1.0,
               label=f"ε floor = {eps:.0e}")
    ax.set_xlabel("log₁₀(avg_employment × E_TOTPOP + ε)", fontsize=10)
    ax.set_ylabel("Number of jurisdictions", fontsize=10)
    ax.set_title("(b) Reference burden-indicator denominator distribution", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_paths = [
        FIG_DIR / "fig_03_qcew_denominator.png",
        WORKTREE_FIG_DIR / "fig_03_qcew_denominator.png",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    print("Generating audit figures...")
    print()
    print("[1/2] Specification curve")
    make_specification_curve()
    print()
    print("[2/2] QCEW denominator behavior")
    make_qcew_denominator_figure()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
