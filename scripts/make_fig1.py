"""make_fig1, AToP framework conceptual figure (Flavor A).

Three-box left-to-right layout:
  INPUT (patient sequence → Transformer → per-token IG)
  → AToP FRAMEWORK (4-stage internal mechanism)
  → AToP EXPLANATIONS (the two novel pattern-level outputs:
                        co-occurrence and cross-visit)

The right-most "AToP Explanations" box is the visual emphasis,
it is what reviewers should walk away thinking "this is what AToP
produces". The middle box stays for the "how it works" mechanism.
The figure has no embedded title (lives in the manuscript caption).

Output: manuscript/jamia_submission/figures/Figure1.png

Usage:
    python atop_package/scripts/make_fig1.py
    python atop_package/scripts/make_fig1.py --preview  # saves to /tmp
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


mpl.rcParams["font.family"] = "sans-serif"
# DejaVu Sans first, it has the unicode glyphs we need (▸, →, ≥, Σ, etc.)
# that Helvetica/Arial often lack on macOS.
mpl.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Helvetica", "Arial",
]


# ── Color palette (muted, paper-friendly) ─────────────────────────────
INPUT_FACE      = "#F4F8FB"
INPUT_EDGE      = "#476582"
FRAMEWORK_FACE  = "#EDF2FA"
FRAMEWORK_EDGE  = "#3F75B5"
EXPLAINS_FACE   = "#FFF5E6"
EXPLAINS_EDGE   = "#D17F1F"
TEXT_DARK       = "#222222"
TEXT_MUTED      = "#555555"
SUBBOX_EDGE     = "#BBBBBB"


def _round_box(ax, x, y, w, h, *, face, edge, lw=1.4):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(p)


def _arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color,
                          mutation_scale=18),
    )


def build_figure(out_path: str) -> None:
    FIG_W, FIG_H = 14, 5  # inches
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ── Box geometry ──────────────────────────────────────────────
    y_bot = 0.4
    h     = 4.2

    ix, iw = 0.4, 2.6                             # INPUT
    fx     = ix + iw + 0.5                        # FRAMEWORK
    fw     = 4.2
    ex     = fx + fw + 0.5                        # EXPLANATIONS
    ew     = FIG_W - ex - 0.4                     # take all remaining width

    # ── INPUT box ─────────────────────────────────────────────────
    _round_box(ax, ix, y_bot, iw, h, face=INPUT_FACE, edge=INPUT_EDGE)
    ax.text(ix + iw / 2, y_bot + h - 0.3, "INPUT",
            fontsize=12, fontweight="bold", color=INPUT_EDGE,
            ha="center", va="top")

    ax.text(ix + iw / 2, y_bot + h - 0.95,
            "Patient EHR sequence\n(diagnoses, procedures,\ndrugs across visits)",
            fontsize=9.5, ha="center", va="top", color=TEXT_DARK)

    ax.text(ix + iw / 2, y_bot + h - 2.05, "↓",
            fontsize=18, ha="center", va="top", color=INPUT_EDGE)

    ax.text(ix + iw / 2, y_bot + h - 2.45,
            "Trained Transformer\n(predicts ŷ)",
            fontsize=10, fontweight="bold", ha="center", va="top",
            color=TEXT_DARK)

    ax.text(ix + iw / 2, y_bot + h - 3.25, "↓",
            fontsize=18, ha="center", va="top", color=INPUT_EDGE)

    ax.text(ix + iw / 2, y_bot + h - 3.65,
            "Per-token Integrated\nGradients (IG)",
            fontsize=9.5, ha="center", va="top", color=TEXT_DARK)

    # arrow INPUT → FRAMEWORK
    _arrow(ax, ix + iw + 0.04, y_bot + h / 2, fx - 0.04, y_bot + h / 2)

    # ── FRAMEWORK box ─────────────────────────────────────────────
    _round_box(ax, fx, y_bot, fw, h, face=FRAMEWORK_FACE,
                edge=FRAMEWORK_EDGE)
    ax.text(fx + fw / 2, y_bot + h - 0.3, "AToP FRAMEWORK",
            fontsize=12, fontweight="bold", color=FRAMEWORK_EDGE,
            ha="center", va="top")
    ax.text(fx + fw / 2, y_bot + h - 0.7, "(internal mechanism)",
            fontsize=9, color=FRAMEWORK_EDGE,
            ha="center", va="top", style="italic")

    stages = [
        ("1. Salient-token selection",
         "retain top-80% IG mass per patient"),
        ("2. Episode mining",
         "frequent token sequences, within and across visits"),
        ("3. Carrier-conditional Σ IG scoring",
         "mean attribution among carriers (token + pattern level)"),
        ("4. Jaccard deduplication",
         "j ≥ 0.2 on test-cohort carrier sets"),
    ]
    y = y_bot + h - 1.3
    for stage, desc in stages:
        ax.text(fx + 0.2, y, stage,
                fontsize=10, fontweight="bold", ha="left", va="top",
                color=TEXT_DARK)
        ax.text(fx + 0.5, y - 0.32, desc,
                fontsize=8.5, ha="left", va="top",
                color=TEXT_MUTED, style="italic")
        y -= 0.75

    # arrow FRAMEWORK → EXPLANATIONS
    _arrow(ax, fx + fw + 0.04, y_bot + h / 2, ex - 0.04, y_bot + h / 2)

    # ── EXPLANATIONS box (the star) ───────────────────────────────
    _round_box(ax, ex, y_bot, ew, h, face=EXPLAINS_FACE,
                edge=EXPLAINS_EDGE, lw=1.8)
    ax.text(ex + ew / 2, y_bot + h - 0.3, "AToP EXPLANATIONS",
            fontsize=13, fontweight="bold", color=EXPLAINS_EDGE,
            ha="center", va="top")
    ax.text(ex + ew / 2, y_bot + h - 0.7,
            "(novel pattern-level outputs)",
            fontsize=9, color=EXPLAINS_EDGE,
            ha="center", va="top", style="italic")

    # ── Co-occurrence row ──
    co_y = y_bot + h - 1.4
    ax.text(ex + 0.2, co_y, "▸ Co-occurrence patterns",
            fontsize=11, fontweight="bold", ha="left", va="top",
            color=TEXT_DARK)
    ax.text(ex + 0.5, co_y - 0.35,
            "single-visit multi-token bundles",
            fontsize=9, ha="left", va="top",
            color=TEXT_MUTED, style="italic")
    # example sub-box
    sb_x = ex + 0.5
    sb_w = ew - 1.0
    _round_box(ax, sb_x, co_y - 1.4, sb_w, 0.62,
                face="white", edge=SUBBOX_EDGE, lw=0.8)
    ax.text(sb_x + 0.2, co_y - 1.10,
            "{Severe MDD; SI}",
            fontsize=10, fontweight="bold", ha="left", va="center",
            color=TEXT_DARK)
    ax.text(sb_x + sb_w - 0.2, co_y - 1.10,
            "Σ IG = +2.17    (risk)",
            fontsize=10, ha="right", va="center", color="#A02020",
            fontweight="bold")

    # ── Cross-visit row ──
    cv_y = co_y - 1.95
    ax.text(ex + 0.2, cv_y, "▸ Cross-visit patterns",
            fontsize=11, fontweight="bold", ha="left", va="top",
            color=TEXT_DARK)
    ax.text(ex + 0.5, cv_y - 0.35,
            "multi-visit trajectories (canonical AToP claim)",
            fontsize=9, ha="left", va="top",
            color=TEXT_MUTED, style="italic")
    _round_box(ax, sb_x, cv_y - 1.4, sb_w, 0.62,
                face="white", edge=SUBBOX_EDGE, lw=0.8)
    ax.text(sb_x + 0.2, cv_y - 1.10,
            "{MDD; SI} → {Recurrent severe MDD; SI}",
            fontsize=10, fontweight="bold", ha="left", va="center",
            color=TEXT_DARK)
    ax.text(sb_x + sb_w - 0.2, cv_y - 1.10,
            "Σ IG = +0.80    (risk)",
            fontsize=10, ha="right", va="center", color="#A02020",
            fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    sz_kb = os.path.getsize(out_path) // 1024
    print(f"Wrote {out_path} ({sz_kb} KB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview", action="store_true",
                    help="Save to /tmp/Figure1_preview.png instead of "
                         "the manuscript figures dir.")
    p.add_argument("--out", default=None,
                    help="Explicit output path.")
    args = p.parse_args()

    if args.out:
        out = args.out
    elif args.preview:
        out = "/tmp/Figure1_preview.png"
    else:
        out = os.path.join(os.path.dirname(__file__), "..", "..",
                            "manuscript", "jamia_submission",
                            "figures", "Figure1.png")
    build_figure(out)


if __name__ == "__main__":
    main()
