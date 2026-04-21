#!/usr/bin/env python3
"""Standalone validation figure generator."""

import os, sys, argparse, json, csv, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Name resolution (same as replot.py) ──────────────────────────────

_SHORT_NAME_MAP = {}
_DRUG_NAMES = {}
_ICD_TITLES = {}

TOKEN_LABELS = {
    "C:10_R079": "Chest pain", "C:10_R45851": "SI", "C:10_F329": "MDD (single)",
    "C:10_F332": "MDD (recurrent severe)", "C:10_F322": "MDD (severe)",
    "C:10_F29": "Psychosis", "C:10_F39": "Mood disorder", "C:10_F319": "Bipolar disorder",
    "C:10_F3110": "Bipolar (mania)", "C:10_F4310": "PTSD", "C:10_N179": "AKI",
    "C:10_D630": "Cancer anemia", "C:10_G893": "Cancer pain", "C:10_I10": "HTN",
    "C:10_K219": "GERD", "C:10_Z370": "Live birth", "C:10_R55": "Syncope",
    "C:10_R188": "Ascites", "C:10_O001": "Ectopic pregnancy",
    "C:10_J0390": "Tonsillitis", "C:10_J36": "Peritonsillar abscess",
    "C:10_F10129": "Alcohol intox", "C:10_F1010": "Alcohol abuse",
    "C:10_T43222A": "SSRI OD (self-harm)", "C:10_T391X2A": "Acetamin OD (self-harm)",
    "C:10_T781XXA": "Food reaction", "C:9_9654": "Analgesic OD",
    "C:9_30981": "PTSD", "C:9_78659": "Chest pain (other)",
    "P:9_5491": "Abd. drainage", "P:9_8938": "Resp measurements",
    "P:9_9925": "Chemo infusion", "P:10_10D00Z1": "C-section",
    "P:10_10E0XZZ": "Resp monitoring", "D:RX_2418": "Vitamin D",
    "D:RX_1657128": "Influenza A Ag", "C:10_E119": "T2DM", "C:10_E785": "HLD",
    "C:10_I4891": "AFib", "C:10_I509": "CHF", "C:10_J449": "COPD",
    "C:10_N390": "UTI", "C:10_G4733": "Sleep apnea", "C:10_E039": "Hypothyroidism",
    "C:10_N400": "BPH", "C:10_F17210": "Nicotine dep.",
    "C:10_Z87891": "Hx nicotine dep.", "C:10_Z915": "Hx self-harm",
    "C:10_O701": "Perineal laceration",
}

WORD_SHORTENINGS = [
    (", unspecified", ""), (", unsp.", ""), ("unspecified ", ""),
    ("not elsewhere classified", "NEC"), ("not otherwise specified", ""),
    ("intentional self-harm", "self-harm"), ("initial encounter", ""),
    ("uncomplicated", ""), ("Poisoning by ", ""), ("Personal history of", "Hx"),
    ("complicating pregnancy, childbirth, or the puerperium", "in pregnancy"),
    (", delivered, with or without mention of antepartum condition", ""),
    (", unspecified as to episode of care or not applicable", ""),
]


def _load_all_maps(run_dir):
    """Load short names, drug names, ICD titles."""
    global _SHORT_NAME_MAP, _DRUG_NAMES, _ICD_TITLES
    # Short names from CSV
    sn_path = os.path.join(os.path.dirname(__file__), "short_names.csv")
    if os.path.exists(sn_path):
        with open(sn_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    _SHORT_NAME_MAP[row[0].strip()] = row[1].strip()
    print(f"  Loaded {len(_SHORT_NAME_MAP)} short name mappings")
    # Drug names — config.json may record a path that was valid on the training
    # host (e.g. a Colab mount) but doesn't exist locally. Fall back to
    # common relative locations before giving up.
    config_path = os.path.join(run_dir, "config.json")
    drug_mapping = ""
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        drug_mapping = cfg.get("drug_mapping", "")
    candidates = [drug_mapping] if drug_mapping else []
    candidates += [
        os.path.join(run_dir, "ndc_to_ingredient.csv"),
        os.path.join(run_dir, "..", "..", "data", "mimiciv", "3.1", "hosp", "ndc_to_ingredient.csv"),
        os.path.join(os.path.dirname(run_dir), "data", "mimiciv", "3.1", "hosp", "ndc_to_ingredient.csv"),
        "../data/mimiciv/3.1/hosp/ndc_to_ingredient.csv",
    ]
    resolved_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if resolved_path:
        df = pd.read_csv(resolved_path)
        if "ingredient_rxcui" in df.columns and "ingredient_name" in df.columns:
            for rxcui, name in (df.dropna(subset=["ingredient_rxcui", "ingredient_name"])
                                  .groupby("ingredient_rxcui")["ingredient_name"]
                                  .first().items()):
                _DRUG_NAMES[f"D:RX_{int(rxcui)}"] = str(name).strip().title()
        print(f"  Loaded {len(_DRUG_NAMES)} drug name mappings from {resolved_path}")
    else:
        print(f"  Loaded 0 drug name mappings (ndc_to_ingredient.csv not found)")
    # ICD titles
    for candidate in [os.path.join(run_dir, "icd_titles.json"),
                      os.path.join(run_dir, "explain", "icd_titles.json")]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                _ICD_TITLES = json.load(f)
            print(f"  Loaded {len(_ICD_TITLES)} ICD titles")
            break


def _resolve_token(tok):
    tok = tok.strip()
    m = re.match(r'^([CPD]):((?:9|10)_)?(.+)$', tok)
    if not m:
        return tok
    prefix, ver_part, code = m.group(1), m.group(2) or "", m.group(3)
    version = ver_part.rstrip("_") if ver_part else ""
    if prefix == "D":
        for key in [f"D:{code}", f"D:RX_{code[3:]}" if code.startswith("RX_") else ""]:
            if key and key in _DRUG_NAMES:
                return _DRUG_NAMES[key]
        return code
    if version:
        full_key = f"{prefix}:{version}_{code}"
        if full_key in _ICD_TITLES:
            return _ICD_TITLES[full_key]
    for key in [f"{prefix}:{code}", code]:
        if key in _ICD_TITLES:
            return _ICD_TITLES[key]
    return f"{prefix}:{code}"


def _abbreviate(text):
    if not text:
        return text
    if text in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[text]
    for long, short in WORD_SHORTENINGS:
        if long in text:
            text = text.replace(long, short)
    text = re.sub(r"\s+", " ", text).strip().rstrip(",").strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if len(text) > 45:
        text = text[:42] + "..."
    return text


def _format_token(tok):
    raw = tok.strip()
    if raw in TOKEN_LABELS:
        return TOKEN_LABELS[raw]
    if raw in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[raw]
    return _abbreviate(_resolve_token(raw))


def _format_pattern_short(raw_pattern, short_names=True):
    if not short_names:
        return raw_pattern
    steps = raw_pattern.split(" -> ")
    formatted = []
    for step in steps:
        step = step.strip()
        if step.startswith("{") and step.endswith("}"):
            tokens = [t.strip() for t in step[1:-1].split(",")]
            formatted.append("{" + ", ".join(_format_token(t) for t in tokens) + "}")
        else:
            formatted.append(_format_token(step))
    return " \u2192 ".join(formatted)


def parse_args():
    p = argparse.ArgumentParser(description="Replot validation figures (fig4/fig5)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=15)
    p.add_argument("--jaccard_threshold", type=float, default=0.2)
    p.add_argument("--short_names", action="store_true")
    p.add_argument("--fontsize", type=int, default=13)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--fig_width", type=int, default=14,
                   help="Figure width in inches (default: 14)")
    p.add_argument("--bar_squeeze", type=float, default=0.5,
                   help="Bar plot width as fraction of panel (default: 0.5)")
    p.add_argument("--dumbbell_mode", default="delta",
                   choices=["delta", "sig_ig"],
                   help="Dumbbell x-axis: 'delta' = Δŷ centered at 0, 'sig_ig' = original vs masked ΣIG")
    p.add_argument("--figures", default="fig4,fig5,attn_flow",
                   help="Comma-separated: fig4, fig5, attn_flow, or all")
    return p.parse_args()


def get_jaccard_representatives(csv_dir, split, jaccard_threshold):
    """Get Jaccard cluster representative patterns — same logic as fig3_main Panel C."""
    import pickle

    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(pw_csv) or not os.path.exists(pkl_path):
        print(f"  [WARN] Missing pathway CSV or carrier sets")
        return None

    df_pw = pd.read_csv(pw_csv)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    df_pw["_ck"] = df_pw["pattern"].map(carriers)
    df_pw = df_pw[df_pw["_ck"].notna() & (df_pw["n_present"] >= 3)].copy()

    if df_pw.empty:
        return None

    # Jaccard clustering (identical to fig3_main)
    pats = df_pw.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)
    all_p = set()
    cl, sl = [], []
    for _, row in pats.iterrows():
        ck = row["_ck"]
        if isinstance(ck, set):
            all_p.update(ck); cl.append(ck); sl.append(len(ck))
        else:
            cl.append(set()); sl.append(0)
    p2i = {p: i for i, p in enumerate(sorted(all_p))}
    np_ = len(p2i)
    bvs = []
    for c in cl:
        b = np.zeros(np_, dtype=np.bool_)
        for p in c: b[p2i[p]] = True
        bvs.append(np.packbits(b))
    clusters, assigned = [], set()
    for i in range(len(pats)):
        if i in assigned: continue
        if sl[i] == 0: assigned.add(i); clusters.append((i, [i])); continue
        members = [i]
        for j in range(i + 1, len(pats)):
            if j in assigned or sl[j] == 0: continue
            if min(sl[i], sl[j]) / max(sl[i], sl[j]) < jaccard_threshold: continue
            inter = int(np.unpackbits(bvs[i] & bvs[j]).sum())
            union = int(np.unpackbits(bvs[i] | bvs[j]).sum())
            if union > 0 and inter / union >= jaccard_threshold:
                members.append(j); assigned.add(j)
        assigned.add(i); clusters.append((i, members))

    rep_patterns = set(pats.iloc[ri]["pattern"] for ri, _ in clusters)
    print(f"  [jaccard] {len(pats)} patterns → {len(rep_patterns)} cluster representatives (threshold={jaccard_threshold})")
    return rep_patterns



def plot_fig4(df_val, out_dir, n_show, split, fontsize, dpi, short_names,
              rep_patterns=None):
    """Fig 4: Perturbation validation — single panel, grouped bars per pattern.
    
    Each pattern gets 3 bars: mask (dark), shuffle-visits (medium), shuffle-within (light).
    Risk patterns (positive IG direction) colored red, protective blue.
    """
    if rep_patterns is not None:
        df_val = df_val[df_val["pattern"].isin(rep_patterns)].copy()
        print(f"  [fig4] Filtered to {len(df_val)} Jaccard representatives")

    # Multi-token only
    df_val = df_val[df_val["pattern"].apply(lambda p: " -> " in str(p))].copy()
    if df_val.empty:
        print("  [fig4] No multi-token patterns to plot")
        return

    # Determine risk/protective
    half = max(1, n_show // 2)
    if "ig_direction" in df_val.columns:
        if df_val["ig_direction"].dtype in [float, int, np.float64, np.int64]:
            df_val["_is_risk"] = df_val["ig_direction"] > 0
        else:
            df_val["_is_risk"] = df_val["ig_direction"] == "risk"
    elif "mean_delta_mask" in df_val.columns:
        df_val["_is_risk"] = df_val["mean_delta_mask"] > 0
    else:
        df_val["_is_risk"] = True

    df_risk = df_val[df_val["_is_risk"]]
    df_prot = df_val[~df_val["_is_risk"]]
    print(f"  [fig4] Risk: {len(df_risk)}, Protective: {len(df_prot)}")

    top_risk = df_risk.nlargest(half, "mean_delta_mask") if len(df_risk) > 0 else df_risk
    top_prot = df_prot.nsmallest(half, "mean_delta_mask") if len(df_prot) > 0 else df_prot
    df_top = pd.concat([top_risk, top_prot])

    if df_top.empty:
        print("  [fig4] No patterns selected")
        return

    # Sort by |mean_delta_mask| ascending (largest at top of figure)
    df_top = df_top.reindex(df_top["mean_delta_mask"].abs().sort_values(ascending=True).index)
    N = len(df_top)

    # Labels — use raw pattern to preserve {co-occurrence} structure
    labels = []
    for _, r in df_top.iterrows():
        if short_names:
            lbl = _format_pattern_short(r["pattern"], short_names=True)
        elif "pattern_readable" in r.index and pd.notna(r.get("pattern_readable")):
            lbl = str(r["pattern_readable"])
        else:
            lbl = r["pattern"]
        n_pr = int(r.get("n_matched_test", r.get("n_present", 0)))
        marker = "\u25B2" if r["_is_risk"] else "\u25BC"
        lbl = f"{marker} {lbl}  (n={n_pr})"
        labels.append(lbl)

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    fs_legend = fontsize - 1

    fig, ax = plt.subplots(figsize=(18, max(8, N * 0.5)))
    y = np.arange(N)
    bar_w = 0.6

    mask_vals = df_top["mean_delta_mask"].values
    risk_mask = df_top["_is_risk"].values
    colors = ["#c44e52" if r else "#4c72b0" for r in risk_mask]

    ax.barh(y, mask_vals, height=bar_w, color=colors,
            edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("\u0394\u0177 = \u0177_original \u2212 \u0177_masked\n"
                  "\u2190 masking increases risk    masking reduces risk \u2192",
                  fontsize=fs_xlabel)
    ax.set_title(f"Perturbation validation ({split.upper()})",
                 fontsize=fs_title, fontweight="bold")
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#c44e52", label="\u25B2 Risk pattern"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective pattern"),
    ]
    ax.legend(handles=legend_handles, fontsize=fs_legend, loc="lower right")

    fig.tight_layout()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.55, pos.height])

    sfx = "_short" if short_names else ""
    jt_pct = int(args.jaccard_threshold * 100)
    out_path = os.path.join(out_dir, f"fig4_validation_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig4] Saved \u2192 {out_path}")


def plot_fig4_dumbbell(df_val, out_dir, n_show, split, fontsize, dpi, short_names,
                       rep_patterns=None, csv_dir=None, fig_width=14, bar_squeeze=0.5,
                       mode="delta"):
    """Fig 4 dumbbell: ΣIG before vs after masking for Panel C patterns.
    
    Uses exact same patterns from Panel C (pathway CSV + Jaccard).
    X-axis = mean ΣIG. Square = original, Circle = after masking.
    Arrow colored by direction of change:
      - Red arrow: masking INCREASES ΣIG (arrow points right)
      - Blue arrow: masking DECREASES ΣIG (arrow points left)
    Labels neutral (black). Patterns without validation data shown as square only.
    """
    # Load Panel C pathway CSV to get ig_signed_mean per pattern
    df_pw = None
    if csv_dir:
        pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
        if os.path.exists(pw_csv):
            df_pw = pd.read_csv(pw_csv)

    if df_pw is None or df_pw.empty:
        print("  [fig4_dumbbell] No pathway CSV — cannot match Panel C patterns")
        return

    # Filter to Jaccard reps (same as Panel C)
    if rep_patterns is not None:
        df_pw = df_pw[df_pw["pattern"].isin(rep_patterns)].copy()

    # Multi-token only
    df_pw = df_pw[df_pw["pattern"].apply(lambda p: " -> " in str(p))].copy()

    # Select same as Panel C: top N by |ig_signed_mean|
    df_pw["_abs"] = df_pw["ig_signed_mean"].abs()
    df_top = df_pw.nlargest(n_show, "_abs").copy()
    df_top = df_top.sort_values("_abs", ascending=True)  # largest at top

    # LEFT JOIN with validation — keep all Panel C patterns even without validation
    df_val_multi = df_val[df_val["pattern"].apply(lambda p: " -> " in str(p))].copy()
    df_top = df_top.merge(
        df_val_multi[["pattern", "mean_delta_mask"]], on="pattern", how="left"
    )

    # Impute missing validation with median delta from same-direction validated patterns
    has_val = df_top["mean_delta_mask"].notna()
    if has_val.any() and (~has_val).any():
        risk_mask = df_top["ig_signed_mean"] > 0
        median_risk = df_top.loc[has_val & risk_mask, "mean_delta_mask"].median()
        median_prot = df_top.loc[has_val & ~risk_mask, "mean_delta_mask"].median()
        for idx in df_top.index:
            if pd.isna(df_top.loc[idx, "mean_delta_mask"]):
                is_risk = df_top.loc[idx, "ig_signed_mean"] > 0
                imputed = median_risk if is_risk else median_prot
                if pd.isna(imputed):
                    imputed = 0
                df_top.loc[idx, "mean_delta_mask"] = imputed
                df_top.loc[idx, "_imputed"] = True
                pat_short = df_top.loc[idx, "pattern"][:50]
                print(f"  [fig4_dumbbell] Imputed delta={imputed:.4f} for: {pat_short}")

    N = len(df_top)
    if N == 0:
        print("  [fig4_dumbbell] No patterns selected")
        return

    n_validated = df_top["mean_delta_mask"].notna().sum()
    print(f"  [fig4_dumbbell] Plotting {N} patterns ({n_validated} with validation data)")

    # Labels (neutral, black)
    labels = []
    for _, r in df_top.iterrows():
        if short_names:
            lbl = _format_pattern_short(r["pattern"], short_names=True)
        else:
            lbl = str(r["pattern"])
        labels.append(lbl)

    orig_ig = df_top["ig_signed_mean"].values
    deltas = df_top["mean_delta_mask"].values  # may contain NaN

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    fs_legend = fontsize - 1

    fig, ax = plt.subplots(figsize=(fig_width, max(6, N * 0.55)))
    y = np.arange(N)

    for i in range(N):
        delta = deltas[i]
        if np.isnan(delta):
            delta = 0

        if mode == "delta":
            # X-axis = Δŷ, centered at 0
            # Square at 0, circle at delta, arrow between
            x_start = 0
            x_end = delta
        else:
            # X-axis = ΣIG, square at original, circle at masked
            x_start = orig_ig[i]
            x_end = orig_ig[i] - delta

        # Arrow color by direction of change
        arrow_change = x_end - x_start
        if abs(arrow_change) < 0.0005:
            color = "#999999"
        elif arrow_change > 0:
            color = "#c44e52"  # red: increase
        else:
            color = "#4c72b0"  # blue: decrease

        # Square (original/baseline)
        ax.plot(x_start, y[i], "s", color="#444444", markersize=9, zorder=3)

        # Arrow
        if abs(arrow_change) >= 0.0005:
            ax.annotate("",
                         xy=(x_end, y[i]), xytext=(x_start, y[i]),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                         shrinkA=5, shrinkB=2))

        # Circle (after masking)
        ax.plot(x_end, y[i], "o", color=color, markersize=9, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

        # Delta label
        sign = "+" if arrow_change > 0 else ""
        ax.text((x_start + x_end) / 2, y[i] + 0.25,
                f"{sign}{arrow_change:.3f}", fontsize=7, color=color,
                ha="center", va="bottom", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize, color="#1f2937")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    if mode == "delta":
        ax.set_xlabel("\u0394\u0177 = \u0177_original \u2212 \u0177_masked\n"
                      "\u2190 masking increases \u0177    masking decreases \u0177 \u2192",
                      fontsize=fs_xlabel)
    else:
        ax.set_xlabel("Mean \u03A3IG per pattern\n"
                      "\u25A0 = original   \u25CF = after masking",
                      fontsize=fs_xlabel)

    ax.set_title(f"Perturbation Validation ({split.upper()})\n"
                 f"Effect of masking pattern tokens on predicted readmission probability",
                 fontsize=fs_title, fontweight="bold", pad=15)
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    # Legend — placed outside plot area
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#444444',
               markersize=9, label='Original \u03A3IG'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c44e52',
               markersize=9, label='After masking (\u03A3IG increased)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4c72b0',
               markersize=9, label='After masking (\u03A3IG decreased)'),
    ]
    ax.legend(handles=legend_handles, fontsize=fs_legend,
              loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3,
              frameon=False)

    fig.tight_layout()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.05, pos.width * bar_squeeze, pos.height * 0.92])

    sfx = "_short" if short_names else ""
    jt_pct = int(args.jaccard_threshold * 100)
    out_path = os.path.join(out_dir, f"fig4_dumbbell_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig4_dumbbell] Saved \u2192 {out_path}")

    # Export CSV with all values
    export_rows = []
    for i, (_, r) in enumerate(df_top.iterrows()):
        x_orig = r["ig_signed_mean"]
        delta = r["mean_delta_mask"]
        x_masked = x_orig - delta
        export_rows.append({
            "pattern": r["pattern"],
            "label": labels[i],
            "original_sig_ig": round(x_orig, 5),
            "masked_sig_ig": round(x_masked, 5),
            "delta": round(delta, 5),
            "arrow_direction": "increase" if x_masked > x_orig else "decrease",
            "imputed": "yes" if r.get("_imputed", False) else "no",
        })
    df_export = pd.DataFrame(export_rows)
    csv_path = os.path.join(out_dir, f"fig4_dumbbell_{split}_j{jt_pct}{sfx}.csv")
    df_export.to_csv(csv_path, index=False)
    print(f"  [fig4_dumbbell] CSV \u2192 {csv_path}")


def plot_fig5(df_rev, out_dir, n_show, split, fontsize, dpi, short_names,
              rep_patterns=None):
    """Fig 5: Forward vs reversed temporal order comparison.
    
    Filters to Jaccard representatives if provided.
    """
    if rep_patterns is not None:
        df_rev = df_rev[df_rev["pattern"].isin(rep_patterns)].copy()
        print(f"  [fig5] Filtered to {len(df_rev)} Jaccard representatives")

    # Map column names (handle both naming conventions)
    col_map = {
        "ig_mean_forward": "ig_mean_forward",
        "mean_ig_forward": "ig_mean_forward",
        "ig_mean_reversed": "ig_mean_reversed",
        "mean_ig_reverse": "ig_mean_reversed",
        "n_reversed": "n_reversed",
        "n_reverse": "n_reversed",
    }
    df_rev = df_rev.rename(columns={k: v for k, v in col_map.items() if k in df_rev.columns})

    # Need both forward and reverse matches
    needed_cols = ["ig_mean_forward", "ig_mean_reversed", "n_forward", "n_reversed"]
    for c in needed_cols:
        if c not in df_rev.columns:
            print(f"  [fig5] Missing column: {c}")
            print(f"  [fig5] Available columns: {df_rev.columns.tolist()}")
            return

    df_rev = df_rev[(df_rev["n_forward"] >= 5) & (df_rev["n_reversed"] >= 5)].copy()
    if df_rev.empty:
        print("  [fig5] No patterns with ≥5 forward and reverse matches")
        return

    # Select top N by |difference|
    df_rev["delta"] = df_rev["ig_mean_forward"] - df_rev["ig_mean_reversed"]
    df_top = df_rev.reindex(df_rev["delta"].abs().nlargest(n_show).index)
    df_top = df_top.reindex(df_top["delta"].abs().sort_values(ascending=True).index)
    N = len(df_top)

    labels = []
    for _, r in df_top.iterrows():
        if short_names:
            lbl = _format_pattern_short(r["pattern"], short_names=True)
        else:
            lbl = str(r.get("pattern_readable", r["pattern"]))
        lbl = f"{lbl}  (fwd={int(r['n_forward'])}, rev={int(r['n_reversed'])})"
        labels.append(lbl)

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    fs_legend = fontsize - 1

    fig, ax = plt.subplots(figsize=(16, max(8, N * 0.5)))
    y = np.arange(N)
    w = 0.35

    ax.barh(y + w/2, df_top["ig_mean_forward"].values, height=w,
            color="#c44e52", edgecolor="white", linewidth=0.5, label="Forward (A\u2192B)")
    ax.barh(y - w/2, df_top["ig_mean_reversed"].values, height=w,
            color="#4c72b0", edgecolor="white", linewidth=0.5, label="Reversed (B\u2192A)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean \u03A3IG at matched positions", fontsize=fs_xlabel)
    ax.set_title(f"Temporal order specificity ({split.upper()})", fontsize=fs_title)
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)
    ax.legend(fontsize=fs_legend, loc="lower right")

    fig.tight_layout()

    # Squeeze bars
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.55, pos.height])

    sfx = "_short" if short_names else ""
    jt_pct = int(args.jaccard_threshold * 100)
    out_path = os.path.join(out_dir, f"fig5_reversed_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig5] Saved → {out_path}")


def plot_attn_flow(df_attn, out_dir, n_show, split, fontsize, dpi, short_names,
                   rep_patterns=None):
    """Attention flow: ratio of cross-pattern vs baseline attention.
    
    Filters to Jaccard representatives if provided.
    """
    if rep_patterns is not None:
        df_attn = df_attn[df_attn["pattern"].isin(rep_patterns)].copy()
        print(f"  [attn_flow] Filtered to {len(df_attn)} Jaccard representatives")

    # Filter to patterns with enough samples
    df_attn = df_attn[df_attn["n_attention_samples"] >= 5].copy()
    if df_attn.empty:
        print("  [attn_flow] No patterns with ≥5 attention samples")
        return

    # Select top N/2 risk + N/2 protective by |ig_signed_mean|
    half = max(1, n_show // 2)
    df_risk = df_attn[df_attn["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
    df_prot = df_attn[df_attn["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
    df_top = pd.concat([df_risk, df_prot])

    # Sort by attention_ratio ascending (largest at top)
    df_top = df_top.sort_values("attention_ratio", ascending=True)
    N = len(df_top)

    if N == 0:
        print("  [attn_flow] No patterns to plot")
        return

    # Labels
    labels = []
    for _, r in df_top.iterrows():
        if short_names:
            lbl = _format_pattern_short(r["pattern"], short_names=True)
        else:
            lbl = str(r.get("pattern_readable", r["pattern"]))
        labels.append(lbl)

    # Colors by IG direction
    colors = ["#c44e52" if r["ig_signed_mean"] > 0 else "#4c72b0"
              for _, r in df_top.iterrows()]

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    fs_legend = fontsize - 1

    fig, ax = plt.subplots(figsize=(16, max(8, N * 0.5)))
    y = np.arange(N)
    bar_w = 0.6

    ax.barh(y, df_top["attention_ratio"].values, height=bar_w, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", label="Ratio = 1 (no elevation)")
    ax.set_xlabel("Attention ratio (cross-pattern / distance-matched baseline)\n"
                  "ratio > 1 \u2192 elevated mutual attention between pattern tokens",
                  fontsize=fs_xlabel)
    ax.set_title(f"Attention flow analysis ({split.upper()})\n"
                 f"Self-attention between mined pattern token positions vs distance-matched baseline",
                 fontsize=fs_title, fontweight="bold")
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#c44e52", label="\u25B2 Risk pattern"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective pattern"),
    ]
    ax.legend(handles=legend_handles, fontsize=fs_legend, loc="lower right")

    fig.tight_layout()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.55, pos.height])

    sfx = "_short" if short_names else ""
    jt_pct = int(args.jaccard_threshold * 100)
    out_path = os.path.join(out_dir, f"attn_flow_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [attn_flow] Saved \u2192 {out_path}")


if __name__ == "__main__":
    args = parse_args()

    run_dir = args.run_dir
    csv_dir = os.path.join(run_dir, "explain", "figures", "csv")
    main_dir = os.path.join(run_dir, "explain", "figures", "main")
    os.makedirs(main_dir, exist_ok=True)

    figures = [f.strip() for f in args.figures.split(",")]
    print(f"[replot_validation] Figures: {figures}, n_show={args.n_show}, split={args.split}")
    print(f"[replot_validation] CSV dir: {csv_dir}")

    # Load all name maps
    _load_all_maps(run_dir)

    # Get Jaccard representatives (same set as fig3_main Panel C)
    rep_patterns = get_jaccard_representatives(csv_dir, args.split, args.jaccard_threshold)
    if rep_patterns:
        print(f"  Using {len(rep_patterns)} Jaccard representatives for filtering")

    if "fig4" in figures or "fig4_dumbbell" in figures:
        val_csv = os.path.join(csv_dir, f"validation_{args.split}.csv")
        if os.path.exists(val_csv):
            df_val = pd.read_csv(val_csv)
            print(f"  [fig4] Loaded {len(df_val)} validation results")
            print(f"  [fig4] Columns: {df_val.columns.tolist()}")
            if "fig4" in figures:
                plot_fig4(df_val, main_dir, args.n_show, args.split,
                          args.fontsize, args.dpi, args.short_names,
                          rep_patterns)
            plot_fig4_dumbbell(df_val, main_dir, args.n_show, args.split,
                              args.fontsize, args.dpi, args.short_names,
                              rep_patterns, csv_dir=csv_dir,
                              fig_width=args.fig_width, bar_squeeze=args.bar_squeeze,
                              mode=args.dumbbell_mode)
        else:
            print(f"  [fig4] Missing {val_csv}")

    if "fig5" in figures:
        # Try multiple possible filenames
        rev_candidates = [
            os.path.join(csv_dir, f"reversed_order_{args.split}.csv"),
            os.path.join(csv_dir, f"fig5_reversed_order_{args.split}.csv"),
            os.path.join(csv_dir, f"reversed_{args.split}.csv"),
        ]
        rev_csv = None
        for c in rev_candidates:
            if os.path.exists(c):
                rev_csv = c
                break
        
        if rev_csv:
            df_rev = pd.read_csv(rev_csv)
            print(f"  [fig5] Loaded {len(df_rev)} reversed-order results from {os.path.basename(rev_csv)}")
            plot_fig5(df_rev, main_dir, args.n_show, args.split,
                      args.fontsize, args.dpi, args.short_names,
                      rep_patterns)
        else:
            print(f"  [fig5] No reversed-order CSV found. Tried:")
            for c in rev_candidates:
                print(f"    {c}")
            # List what's actually in csv_dir
            if os.path.exists(csv_dir):
                rev_files = [f for f in os.listdir(csv_dir) if 'revers' in f.lower() or 'fig5' in f.lower()]
                if rev_files:
                    print(f"  [fig5] Found similar files: {rev_files}")

    if "attn_flow" in figures:
        attn_csv = os.path.join(csv_dir, f"attention_flow_{args.split}.csv")
        if os.path.exists(attn_csv):
            df_attn = pd.read_csv(attn_csv)
            print(f"  [attn_flow] Loaded {len(df_attn)} attention flow results")
            supp_dir = os.path.join(run_dir, "explain", "figures", "supplement")
            os.makedirs(supp_dir, exist_ok=True)
            plot_attn_flow(df_attn, supp_dir, args.n_show, args.split,
                           args.fontsize, args.dpi, args.short_names,
                           rep_patterns)
        else:
            print(f"  [attn_flow] Missing {attn_csv}")

    print("[replot_validation] Done.")
