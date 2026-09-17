"""make_fig2_pptx, Build Figure 2 Panel A (visit timeline) as native
PowerPoint shapes so the author can hand-edit it.

What this writes:
  • Horizontal timeline (line shape)
  • One oval per visit (dots; index visit in orange)
  • A "Day N" label under each dot (light gray)
  • ALL salient tokens per visit, ranked by |IG| descending, each as a
    separate editable text box. Coloring:
        – RED  if the token IS surfaced by one of Panel B's top-N patterns
                AND its IG is positive (risk)
        – BLUE if surfaced AND IG is negative (protective)
        – LIGHT GRAY otherwise (token did not make it into any of the
                top-N pattern bars in Panel B, visually de-emphasized
                so the reader can see WHICH tokens drive Panel B vs
                which are individually small / orphaned)
  • Readmission marker (red dot + label) just right of V_idx
  • Index-visit ŷ badge (orange)

Also writes a sidecar `Figure2_PanelA_text.txt` next to the PPTX,
plain text listing of every visit's tokens (with the same star/gray
markers) for quick copy-paste reference.

Panel B is NOT included, the user is editing it separately.

Usage:
    python scripts/make_fig2_pptx.py --run_dir ../runs/mimiciv_readmission_30d \\
        --mimic_dir ../data/mimiciv/3.1/hosp
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from make_fig2 import (
    abbreviate_label, fetch_visit_times,
    match_patterns_with_positions, greedy_assignment, backfill_net_sigma,
)
from atop.run_layout import RunLayout
from atop.explain.pattern_labels import PatternLabelResolver


# ── Colors ────────────────────────────────────────────────────────────
RISK_RGB   = RGBColor(0xC8, 0x3D, 0x3D)   # red (risk-pushing IG, surfaced)
PROT_RGB   = RGBColor(0x1F, 0x6E, 0xB5)   # blue (protective IG, surfaced)
GRAY_RGB   = RGBColor(0xBB, 0xBB, 0xBB)   # light gray (token NOT in Panel B)
DAY_RGB    = RGBColor(0x99, 0x99, 0x99)
TEXT_RGB   = RGBColor(0x22, 0x22, 0x22)
MUTED_RGB  = RGBColor(0x66, 0x66, 0x66)
INDEX_RGB  = RGBColor(0xE5, 0x9E, 0x2A)   # orange (index-visit emphasis)
LINE_RGB   = RGBColor(0x22, 0x22, 0x22)


# ── PPT helpers ───────────────────────────────────────────────────────
def add_textbox(slide, x_in, y_in, w_in, h_in, text, *,
                fontsize=10, bold=False, italic=False,
                color=TEXT_RGB, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(
        Inches(x_in), Inches(y_in), Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0);  tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(fontsize)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_line(slide, x0_in, y0_in, x1_in, y1_in, color=LINE_RGB, weight_pt=1.4):
    ln = slide.shapes.add_connector(
        1, Inches(x0_in), Inches(y0_in), Inches(x1_in), Inches(y1_in))
    ln.line.color.rgb = color
    ln.line.width = Pt(weight_pt)
    return ln


def add_dot(slide, cx_in, cy_in, radius_in=0.08, color=LINE_RGB):
    d = radius_in * 2
    shp = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(cx_in - radius_in), Inches(cy_in - radius_in),
        Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    shp.line.width = Pt(0.5)
    return shp


# ── Layout config (inches) ────────────────────────────────────────────
SLIDE_W = 16.0
SLIDE_H = 10.0
TIMELINE_LEFT_PAD  = 0.5
TIMELINE_RIGHT_PAD = 2.0     # leave room for Readmission marker
TIMELINE_Y         = 1.30    # timeline near the top
DAY_LABEL_Y_OFFSET = 0.18    # day label gap below the dot
TOKEN_BLOCK_TOP    = TIMELINE_Y + 0.62  # first token row starts here
TOKEN_LINE_DY      = 0.20    # vertical spacing per token row
BODY_FS            = 9
DAY_FS             = 10
COL_W              = 2.7     # text-block column width per visit (inches)


# ── Selective gap compression: halve the single largest gap ───────────
def compute_x_positions_inches(days: np.ndarray, x_left_in: float,
                                 x_right_in: float) -> list:
    day_max = float(max(days[-1], 1.0))
    x = [d / day_max for d in days]
    if len(x) >= 2:
        gaps = [x[i] - x[i-1] for i in range(1, len(x))]
        max_gap_idx = 1 + int(np.argmax(gaps))
        max_gap = x[max_gap_idx] - x[max_gap_idx - 1]
        new_gap = max_gap * 0.5
        delta = max_gap - new_gap
        for j in range(max_gap_idx, len(x)):
            x[j] -= delta
    span = x[-1] - x[0] if x[-1] > x[0] else 1.0
    available = x_right_in - x_left_in
    return [x_left_in + (xi - x[0]) / span * available for xi in x]


# ── Data pipeline ─────────────────────────────────────────────────────
def load_exemplar_data(run_dir: str, mimic_dir: str):
    """Run the same matching + greedy assignment make_fig2 uses, and
    return per-visit token rows (sorted by |IG| desc) annotated with
    whether each position is surfaced by one of Panel B's top-N
    patterns."""
    layout = RunLayout(run_dir)
    with open(layout.exemplars_json) as f:
        exemplar = json.load(f)["main_tp"]
    pid, hadm = exemplar["patient_id"], exemplar["index_hadm_id"]

    # IG
    df_ig_all = pd.read_csv(layout.ig_csv("test"),
                              dtype={"patient_id": str, "index_hadm_id": str})
    df_ig = df_ig_all[(df_ig_all["patient_id"] == pid)
                       & (df_ig_all["index_hadm_id"] == hadm)] \
        .sort_values("pos").reset_index(drop=True)
    y_prob = float(df_ig["y_prob"].iloc[0]) if "y_prob" in df_ig.columns \
        else float(exemplar.get("y_prob", 0.93))
    y_true = int(df_ig["readmission"].iloc[0]) if "readmission" in df_ig.columns \
        else int(exemplar.get("y_true", 1))

    # Salient blocks
    sal = df_ig[df_ig["is_salient"] == True]  # noqa: E712
    salient_visit_blocks = [
        sorted(sal[sal["visit_idx"] == v]["token_str"].unique().tolist())
        for v in sorted(sal["visit_idx"].unique())
    ]

    # Patterns + greedy
    pat_files = sorted(glob.glob(os.path.join(
        layout.mining_dir, "all_scored_*.csv")), key=os.path.getmtime)
    df_patterns = pd.read_csv(pat_files[-1])
    resolver = PatternLabelResolver(layout.run_dir, verbose=False)
    def fmt_pat(s, max_len=95):
        return resolver.format_pattern(s, short_names=True)
    matches = match_patterns_with_positions(
        df_ig, salient_visit_blocks, df_patterns, fmt_pat)
    kept = greedy_assignment(matches)
    ig_signed_by_pos = {int(r["pos"]): float(r["ig_signed"])
                         for _, r in df_ig.iterrows()}
    backfill_net_sigma(kept, ig_signed_by_pos)
    kept = [m for m in kept if abs(m.net_sigma_ig) > 1e-9]
    kept.sort(key=lambda m: -abs(m.net_sigma_ig))

    # Top-12 patterns (Panel B's default cap) → surfaced position set
    TOP_N = 12
    surfaced_pos: set[int] = set()
    surfaced_patterns = []
    for m in kept[:TOP_N]:
        for (pos, _) in m.net_positions:
            surfaced_pos.add(int(pos))
        surfaced_patterns.append(
            {"sigma": m.net_sigma_ig, "label": m.pattern_readable})

    # Visit times
    visit_times, next_adm = fetch_visit_times(pid, hadm, mimic_dir)
    n_visits = int(df_ig["visit_idx"].max()) + 1
    days = visit_times["days_since_v1"].values.astype(float) \
        if visit_times is not None else np.arange(n_visits) * 100.0

    # Per-visit token list, sorted by |IG| desc
    visits = []
    for vi in range(n_visits):
        rows = df_ig[df_ig["visit_idx"] == vi].copy()
        rows["abs_ig"] = rows["ig_signed"].abs()
        rows = rows.sort_values("abs_ig", ascending=False)
        tokens = []
        for _, r in rows.iterrows():
            short = abbreviate_label(r["token_str_readable"])
            if len(short) > 38:
                short = short[:37] + "…"
            tokens.append({
                "ig": float(r["ig_signed"]),
                "label": short,
                "raw": r["token_str_readable"],
                "token_str": r["token_str"],
                "pos": int(r["pos"]),
                "surfaced": int(r["pos"]) in surfaced_pos,
            })
        visits.append({"vi": vi, "day": float(days[vi]), "tokens": tokens})

    return {
        "pid": pid, "hadm": hadm, "y_prob": y_prob, "y_true": y_true,
        "n_visits": n_visits, "visits": visits,
        "next_adm": next_adm,
        "surfaced_patterns": surfaced_patterns,
    }


# ── Build the PPTX ────────────────────────────────────────────────────
def build_pptx(data: dict, out_path: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    n_visits = data["n_visits"]
    idx_x = n_visits - 1
    days = np.array([v["day"] for v in data["visits"]])

    # Header
    add_textbox(slide, 0.25, 0.20, 14.0, 0.35,
                f"A · Visit timeline   ·   pid {data['pid']}   ·   "
                f"hadm {data['hadm']}   ·   ŷ = {data['y_prob']:.2f}",
                fontsize=13, bold=True)
    add_textbox(slide, 0.25, 0.55, 14.0, 0.30,
                "Tokens ranked by |IG| ↓.  Color = surfaced by a Panel B "
                "pattern (red = risk-pushing, blue = protective).  "
                "Light gray = not surfaced.",
                fontsize=9, italic=True, color=MUTED_RGB)

    # x positions
    x_left = TIMELINE_LEFT_PAD
    x_right = SLIDE_W - TIMELINE_RIGHT_PAD
    x_positions = compute_x_positions_inches(days, x_left, x_right)
    last_gap = x_positions[idx_x] - x_positions[idx_x - 1] \
        if n_visits >= 2 else 0.4
    r30_x = x_positions[idx_x] + max(last_gap * 0.5, 0.5)
    r30_x = min(r30_x, SLIDE_W - 1.6)

    # Timeline
    add_line(slide, x_left - 0.2, TIMELINE_Y, r30_x + 0.2, TIMELINE_Y,
              color=LINE_RGB, weight_pt=1.4)

    # Per visit: dot + day label + ALL tokens
    for i, v in enumerate(data["visits"]):
        is_idx = (i == idx_x)
        x = x_positions[i]
        dot_color = INDEX_RGB if is_idx else LINE_RGB
        add_dot(slide, x, TIMELINE_Y,
                 radius_in=0.10 if is_idx else 0.075,
                 color=dot_color)

        # Day label
        add_textbox(slide, x - 0.6, TIMELINE_Y + DAY_LABEL_Y_OFFSET,
                    1.6, 0.30,
                    f"Day {int(v['day'])}",
                    fontsize=DAY_FS, bold=True, color=DAY_RGB,
                    align=PP_ALIGN.CENTER)

        # ŷ badge for the index visit (right under day label)
        token_top = TOKEN_BLOCK_TOP
        if is_idx:
            add_textbox(slide, x - 0.7, TIMELINE_Y + 0.42, 1.8, 0.28,
                        f"ŷ = {data['y_prob']:.2f}",
                        fontsize=11, bold=True, color=INDEX_RGB,
                        align=PP_ALIGN.CENTER)
            token_top = TOKEN_BLOCK_TOP + 0.20

        # ALL tokens, stacked
        for j, t in enumerate(v["tokens"]):
            if t["surfaced"]:
                color = RISK_RGB if t["ig"] > 0 else PROT_RGB
                bold = True
            else:
                color = GRAY_RGB
                bold = False
            y = token_top + j * TOKEN_LINE_DY
            add_textbox(slide,
                        x - COL_W / 2, y,
                        COL_W, TOKEN_LINE_DY,
                        f"{t['ig']:+.2f}  {t['label']}",
                        fontsize=BODY_FS, color=color, bold=bold,
                        align=PP_ALIGN.LEFT)

    # Readmission marker
    r30_color = RISK_RGB if data["y_true"] == 1 else PROT_RGB
    add_dot(slide, r30_x, TIMELINE_Y, radius_in=0.10, color=r30_color)
    if data["y_true"] == 1 and data["next_adm"] is not None:
        r_label = "Readmission"
        r_sub = f"{data['next_adm']['days_post']} days post-discharge"
    elif data["y_true"] == 1:
        r_label, r_sub = "Readmission", "within 30 days"
    else:
        r_label, r_sub = "No readmission", "within 30 days"
    add_textbox(slide, r30_x - 0.1, TIMELINE_Y - 0.55, 2.0, 0.30,
                r_label, fontsize=12, bold=True, color=r30_color)
    add_textbox(slide, r30_x - 0.1, TIMELINE_Y - 0.25, 2.0, 0.25,
                r_sub, fontsize=9, italic=True, color=MUTED_RGB)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prs.save(out_path)
    print(f"  ✓ PPTX  → {out_path} ({os.path.getsize(out_path)//1024} KB)")


# ── Sidecar text dump ─────────────────────────────────────────────────
def write_text_sidecar(data: dict, out_path: str) -> None:
    lines = []
    lines.append(f"Figure 2 Panel A, per-visit token listing")
    lines.append(f"Patient: pid={data['pid']}  hadm={data['hadm']}  "
                  f"ŷ={data['y_prob']:.2f}  y_true={data['y_true']}")
    if data["next_adm"]:
        lines.append(f"Next admission: {data['next_adm']['days_post']} "
                      f"days post-discharge")
    lines.append("")
    lines.append("Legend:  ★ = token is surfaced by a Panel B pattern  "
                  "·  = token not surfaced (would render gray)")
    lines.append("")
    lines.append("Panel B top-12 surfaced patterns (for cross-reference):")
    for i, p in enumerate(data["surfaced_patterns"], 1):
        lines.append(f"  {i:2d}. Σnet={p['sigma']:+.3f}  {p['label']}")
    lines.append("")
    lines.append("=" * 72)
    for v in data["visits"]:
        lines.append(f"V{v['vi']+1}  ·  Day {int(v['day'])}  "
                      f"·  {len(v['tokens'])} tokens")
        for t in v["tokens"]:
            mark = "★" if t["surfaced"] else "·"
            sign = "+" if t["ig"] > 0 else "−"
            lines.append(f"   {mark}  {sign}{abs(t['ig']):.3f}   "
                          f"{t['label']}    [raw: {t['raw']}]")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Text  → {out_path} ({os.path.getsize(out_path)//1024} KB)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--mimic_dir", required=True)
    p.add_argument("--output-path", default=None,
                    help="Default: manuscript/jamia_submission/figures/Figure2.pptx")
    args = p.parse_args()

    print("[1] Loading exemplar + running pattern pipeline…")
    data = load_exemplar_data(args.run_dir, args.mimic_dir)
    print(f"    pid={data['pid']} hadm={data['hadm']} "
          f"n_visits={data['n_visits']} y_prob={data['y_prob']:.2f}")
    for v in data["visits"]:
        n_surf = sum(1 for t in v["tokens"] if t["surfaced"])
        print(f"    V{v['vi']+1}: {len(v['tokens'])} tokens "
              f"({n_surf} surfaced in Panel B)")

    if args.output_path is None:
        out_pptx = os.path.join(os.path.dirname(__file__), "..", "..",
                                  "manuscript", "jamia_submission",
                                  "figures", "Figure2.pptx")
    else:
        out_pptx = args.output_path
    out_txt = out_pptx.replace(".pptx", "_PanelA_text.txt")

    print("[2] Writing outputs…")
    build_pptx(data, out_pptx)
    write_text_sidecar(data, out_txt)


if __name__ == "__main__":
    main()
