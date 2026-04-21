"""
Regenerate Figure 2 (single-patient IG attribution) at 300 DPI from
the cached CSV at runs/full_CPD/explain/figures/csv/fig2_patient_ig_tokens_test.csv.

The cached CSV contains 15 top-salient tokens for one representative test
patient with columns: token_str, token_readable, ig_abs, ig_signed.

Output:
  runs/full_CPD/explain/figures/main/fig2_patient_ig_tokens_test.png  (300 DPI)
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

CSV_DIR = "../runs/full_CPD/explain/figures/csv"
OUT_DIR = "../runs/full_CPD/explain/figures/main"

RISK_COLOR = "#c0392b"      # red for risk-driving (ig > 0)
PROT_COLOR = "#2c7bb6"      # blue for protective  (ig < 0)
AXIS_LW    = 0.8


def plot_one(csv_path: str, out_path: str, title: str):
    df = pd.read_csv(csv_path)
    # Rank by |IG| descending so most salient is at the top of the plot.
    df = df.sort_values("ig_abs", ascending=True).reset_index(drop=True)

    colors = [RISK_COLOR if v > 0 else PROT_COLOR for v in df["ig_signed"]]

    fig, ax = plt.subplots(figsize=(7.1, 5.5))
    ax.barh(range(len(df)), df["ig_signed"].values,
            color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["token_readable"].values, fontsize=10)
    ax.axvline(0, color="black", linewidth=AXIS_LW)

    # Clean spines
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)

    ax.set_xlabel("Integrated Gradients attribution (signed)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8, loc="left")

    # Symmetric x-range around 0 so sign reads cleanly
    xmax = max(df["ig_abs"].max(), 1e-6)
    ax.set_xlim(-xmax * 1.1, xmax * 1.1)
    ax.tick_params(axis="x", labelsize=10)

    legend = [
        mpatches.Patch(color=RISK_COLOR, label="Risk-driving (IG > 0)"),
        mpatches.Patch(color=PROT_COLOR, label="Protective (IG < 0)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = [
        ("fig2_patient_ig_tokens_test.csv",
         "fig2_patient_ig_tokens_test.png",
         "Figure 2. Single-patient attribution (top 15 salient tokens)"),
        ("fig2_patient_ig_tokens_test_ex2.csv",
         "fig2_patient_ig_tokens_test_ex2.png",
         "Figure 2 (ex2). Single-patient attribution (top 15 salient tokens)"),
        ("fig2_patient_ig_tokens_test_ex3.csv",
         "fig2_patient_ig_tokens_test_ex3.png",
         "Figure 2 (ex3). Single-patient attribution (top 15 salient tokens)"),
    ]
    for csv_name, png_name, title in targets:
        csv_path = os.path.join(CSV_DIR, csv_name)
        out_path = os.path.join(OUT_DIR, png_name)
        if not os.path.exists(csv_path):
            print(f"  [skip] missing {csv_path}")
            continue
        saved = plot_one(csv_path, out_path, title)
        print(f"  [saved] {saved}")


if __name__ == "__main__":
    main()
