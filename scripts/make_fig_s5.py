"""make_fig_s5, Supplementary Figure S5: prevalence vs impact scatter.

Two-panel scatter showing that AToP's carrier-averaged |ΣIG| is partially
independent of classical effect-size measures: (A) prevalence vs |ΣIG|
surfaces rare-but-decisive patterns that bulk-prevalence ranking would
miss; (B) relative risk vs |ΣIG| shows the two metrics capture
partially-independent signal (patterns with similar RR span a wide |ΣIG|
range). Uses Relative Risk (RR) consistent with main-paper Tables 3 and
4; Odds Ratio is not used in the manuscript.

Data sources (merged on pattern key):
  - pathway_importance_<split>.csv   IG (signed mean), test-cohort prevalence
  - all_scored_<hash>.csv (mining)   training-cohort readmission counts → RR

The two CSVs encode different pattern enumerations (PW has 39K cross-
visit patterns; SC has 24K all-types). Intersection is ~5K cross-visit
patterns; that's the displayed pool.

Usage:
    python scripts/make_fig_s5.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


RISK = "#c44e52"   # canonical risk red (matches PHEN_COLORS / Fig S1 / S3)
PROT = "#4c72b0"   # canonical protective blue (matches PHEN_COLORS / Fig S1 / S3)


def _short_pattern_label(pat: str, max_len: int = 52) -> str:
    s = re.sub(r"\s*\([^)]*\)", "", str(pat))
    s = s.replace("{", "").replace("}", "").strip()
    if len(s) > max_len:
        s = s[:max_len - 1] + "\u2026"
    return s


def _pick_rare_decisive(df: pd.DataFrame, side: str, n: int) -> pd.DataFrame:
    sub = df[df["sign"] == side].copy()
    med_prev = df["prevalence_pct"].median()
    hi_ig = df["abs_ig"].quantile(0.75)
    rare = sub[(sub["prevalence_pct"] <= med_prev) & (sub["abs_ig"] >= hi_ig)]
    return rare.nlargest(n, "abs_ig")


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S5 (prevalence vs impact)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v1")
    p.add_argument("--split", default="test")
    p.add_argument("--n-annotate-per-side", type=int, default=5,
                   help="How many rare-but-decisive patterns to annotate per side")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS5_prevalence_vs_impact.png")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory holding pathway_importance_*.csv (default: <run_dir>/5_analysis)")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir

    pw_csv = layout.pathway_csv(args.split) if csv_dir == layout.analysis_dir else \
             os.path.join(csv_dir, f"pathway_importance_{args.split}.csv")
    if not os.path.exists(pw_csv):
        # Fall back to legacy filename for back-compat
        legacy = os.path.join(csv_dir, f"fig3_pathway_importance_{args.split}.csv")
        if os.path.exists(legacy):
            pw_csv = legacy
        else:
            sys.exit(f"  [ERROR] Missing {pw_csv}")
    df = pd.read_csv(pw_csv)
    print(f"[1] Loaded {len(df):,} patterns from pathway_importance (test cohort)")

    # ── Merge with mining cache to attach readmission counts ──
    # pathway_importance carries IG + test-cohort prevalence; the raw
    # mining cache carries training-cohort readmission counts needed for
    # Relative Risk. Intersection is the cross-visit patterns present
    # in both pools (~5K).
    import glob as _glob
    scored_csvs = sorted(_glob.glob(os.path.join(layout.mining_dir,
                                                  "all_scored_*.csv")),
                          key=os.path.getmtime)
    if not scored_csvs:
        sys.exit("  [ERROR] No all_scored_*.csv mining cache found "
                  f"under {layout.mining_dir}/")
    sc = pd.read_csv(scored_csvs[-1])
    print(f"    mining cache: {len(sc):,} scored patterns "
           f"({os.path.basename(scored_csvs[-1])})")

    # Compute training-cohort RR per pattern.
    # baseline = positive class fraction in training cohort
    base_pos = float(sc["n_total_readmit1"].iloc[0])
    base_neg = float(sc["n_total_readmit0"].iloc[0])
    baseline = base_pos / (base_pos + base_neg)
    print(f"    training baseline readmit rate: {baseline*100:.2f}%")
    sc["carrier_readmit"] = (sc["n_present_readmit1"]
                              / (sc["n_present_readmit1"]
                                 + sc["n_present_readmit0"]).replace(0, 1))
    sc["relative_risk"] = sc["carrier_readmit"] / baseline

    # Merge on pattern key
    df = df.merge(sc[["pattern", "relative_risk", "n_present_readmit1",
                       "n_present_readmit0"]],
                   on="pattern", how="inner")
    print(f"    after merge (IG + RR available): {len(df):,} patterns")

    # Multi-token only (sanity, intersect is already cross-visit)
    df = df[df["pattern"].apply(lambda p: " -> " in str(p) or "{" in str(p))].copy()
    df["abs_ig"] = df["ig_signed_mean"].abs()
    df["prevalence_pct"] = df["prevalence"] * 100
    df["sign"] = np.where(df["ig_signed_mean"] >= 0, "risk", "protective")
    df["log_n"] = np.log10(df["n_present"].clip(lower=1))
    df["log_rr"] = np.log10(df["relative_risk"].clip(lower=1e-3))
    print(f"    multi-token (final pool): {len(df):,}")

    rare_risk = _pick_rare_decisive(df, "risk", args.n_annotate_per_side)
    rare_prot = _pick_rare_decisive(df, "protective", args.n_annotate_per_side)
    print(f"[2] Annotating {len(rare_risk)} risk + {len(rare_prot)} protective rare-decisive")

    # ── Pattern label resolver (canonical readable names) ──
    from atop.explain.pattern_labels import PatternLabelResolver
    resolver = PatternLabelResolver(
        args.run_dir,
        short_names_csv=os.path.join(os.path.dirname(__file__),
                                       "short_names.csv"),
    )

    def _resolve_label(pat: str, max_len: int = 48) -> str:
        s = resolver.format_pattern(pat, short_names=True)
        s = s.replace("{", "").replace("}", "").strip()
        if len(s) > max_len:
            s = s[:max_len - 1] + "…"
        return s

    # ── Render ────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.8),
                                    gridspec_kw={"wspace": 0.30})

    # ════ Panel A: prevalence vs |ΣIG| ════
    risk_df = df[df["sign"] == "risk"]
    prot_df = df[df["sign"] == "protective"]
    for sub, color, label in [(risk_df, RISK, "Risk (ΣIG > 0)"),
                                (prot_df, PROT, "Protective (ΣIG < 0)")]:
        axA.scatter(sub["prevalence_pct"], sub["abs_ig"],
                     s=8 + 5 * sub["log_n"], alpha=0.40,
                     color=color, edgecolor="none", label=label, zorder=2)

    # Mark the rare-decisive dots with small empty rings (no overlapping
    # text labels), then list the same patterns in a single corner
    # textbox below the panel title. much cleaner than per-dot
    # annotations, which stack on top of each other in the low-
    # prevalence / high-|ΣIG| corner.
    label_pool = pd.concat([rare_risk, rare_prot])
    axA.scatter(label_pool["prevalence_pct"], label_pool["abs_ig"],
                 s=120, facecolors="none",
                 edgecolors=["#7a1f24" if s == "risk" else "#1f3c66"
                              for s in label_pool["sign"]],
                 linewidths=1.6, zorder=5)

    # Compose the corner list. Risk first (red), then protective (blue).
    risk_lines = ["Top rare-decisive risk patterns:"]
    for i, (_, r) in enumerate(rare_risk.iterrows(), 1):
        risk_lines.append(f"  {i}. {_resolve_label(r['pattern'], 46)}  "
                           f"(n={int(r['n_present'])})")
    prot_lines = ["", "Top rare-decisive protective patterns:"]
    for i, (_, r) in enumerate(rare_prot.iterrows(), 1):
        prot_lines.append(f"  {i}. {_resolve_label(r['pattern'], 46)}  "
                           f"(n={int(r['n_present'])})")
    box_text = "\n".join(risk_lines + prot_lines)
    # Listed below the plotting area so no data point is hidden
    # (the former in-axes box covered the high-|ΣIG| tail).
    fig.text(
        0.01, -0.02, box_text,
        ha="left", va="top",
        fontsize=8.5, color="#111111", family="DejaVu Sans",
    )

    axA.set_xscale("log")
    axA.set_xlim(df["prevalence_pct"].min() * 0.7,
                  df["prevalence_pct"].max() * 1.5)
    axA.set_xlabel("Prevalence (% of test patients carrying the pattern)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axA.set_ylabel("Carrier-averaged |ΣIG|",
                    fontsize=12, fontweight="bold", labelpad=8)
    axA.set_title(
        "A · Prevalence vs |ΣIG|. "
        "rare-but-decisive patterns lie low-left",
        fontsize=12, loc="left", pad=10,
    )
    axA.tick_params(labelsize=10)
    axA.grid(True, which="both", color="#eeeeee", linewidth=0.5)
    axA.legend(loc="upper right", fontsize=10, framealpha=0.95)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)

    # ════ Panel B: log(RR) vs |ΣIG| ════
    df_rr = df[df["relative_risk"] > 0].copy()
    risk_rr = df_rr[df_rr["sign"] == "risk"]
    prot_rr = df_rr[df_rr["sign"] == "protective"]
    for sub, color, label in [(risk_rr, RISK, "Risk (ΣIG > 0)"),
                                (prot_rr, PROT, "Protective (ΣIG < 0)")]:
        axB.scatter(sub["log_rr"], sub["abs_ig"],
                     s=8 + 5 * sub["log_n"], alpha=0.40,
                     color=color, edgecolor="none", label=label, zorder=2)

    # Reference vertical line at RR = 1 (baseline / null effect).
    axB.axvline(0, color="#888888", linestyle="--", linewidth=1.0,
                 alpha=0.7, zorder=1)

    rho, _ = spearmanr(df_rr["log_rr"], df_rr["abs_ig"])
    axB.text(
        0.97, 0.97,
        f"Spearman ρ (|ΣIG| vs log RR) = {rho:.2f}\n"
        "RR and |ΣIG| capture partially-independent\n"
        "signal: patterns with similar RR span a wide\n"
        "|ΣIG| range, so AToP surfaces patterns that\n"
        "RR-ranking alone would miss.",
        transform=axB.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec="#aaaaaa", lw=0.6),
        zorder=5,
    )
    axB.set_xlabel(
        "log₁₀(Relative Risk vs cohort baseline)",
        fontsize=12, fontweight="bold", labelpad=8,
    )
    axB.set_ylabel("Carrier-averaged |ΣIG|",
                    fontsize=12, fontweight="bold", labelpad=8)
    axB.set_title(
        "B · |ΣIG| vs log(Relative Risk). "
        "the two metrics disagree",
        fontsize=12, loc="left", pad=10,
    )
    axB.tick_params(labelsize=10)
    axB.grid(True, which="both", color="#eeeeee", linewidth=0.5)
    axB.legend(loc="lower right", fontsize=10, framealpha=0.95)
    for side in ("top", "right"):
        axB.spines[side].set_visible(False)

    # No in-image suptitle; the DOCX caption supplies the figure heading.
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] Saved figure: {args.output_path} ({os.path.getsize(args.output_path)//1024} KB)")

    os.makedirs(csv_dir, exist_ok=True)
    out_csv = os.path.join(csv_dir, "fig_s5_rare_decisive_patterns.csv")
    pd.concat([rare_risk.assign(group="risk"),
               rare_prot.assign(group="protective")])[
        ["group", "pattern_readable", "prevalence_pct", "abs_ig",
         "ig_signed_mean", "relative_risk", "n_present"]
    ].to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")


if __name__ == "__main__":
    main()
