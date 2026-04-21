"""
Redesigned Figure 2 — single-patient attribution.

Replaces the old 4-panel layout (SHAP | IG | chronological tokens | patterns)
with a cleaner 3-panel layout:

  Panel A (top, full width): SHAP vs IG agreement for the union of top-8
    tokens by each method — paired bars on a shared y-axis (normalized so
    the two magnitude scales are visually comparable), with token names on
    the left. Makes method agreement / disagreement immediately visible.

  Panel B (bottom-left): Visit-level attribution timeline. X-axis = visit
    index (V1…Vn); each visit column contains a stacked set of salient
    tokens plotted as small bars colored by signed IG. A gray line traces
    the cumulative signed IG across visits to show where risk accrues
    temporally. Token labels appear inside or beside the bars.

  Panel C (bottom-right): Top-6 matched temporal patterns by |Σ IG|,
    drawn with FULL pattern names (no `…` truncation), OR and carrier
    count annotated at the right end of each bar. Signed IG sum on the
    x-axis.

This script runs standalone — it does NOT require MIMIC tables or the
full explainer.from_bundle() path (which is slow to start from Google
Drive). It only needs:
  runs/full_CPD/model.pt
  runs/full_CPD/vocab.pkl
  runs/full_CPD/config.json
  runs/full_CPD/icd_titles.json
  runs/full_CPD/ig_cache/tensors_test.pt         (from regen_tensors_test.py)
  runs/full_CPD/ig_cache/ig_test.csv             (already on disk)
  runs/full_CPD/ig_cache/sequences_test.pkl      (already on disk)
  runs/full_CPD/mining_cache/all_scored_*.csv    (any one is fine)

Output: ../runs/full_CPD/explain/figures/main/fig2_patient_explanation_test.png
        copied to ../manuscript/jamia_submission/figures/Figure2.png
"""
from __future__ import annotations
import os, sys, json, glob, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.config import PAD_IDX, SPECIAL_TOKENS
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.utils import load_drug_names
from atop.explain.matching import _match_all_patterns


# ── Local label resolver (icd_titles.json is token-keyed, not tuple-keyed) ──
def make_token_formatter(icd_title_by_token: dict, drug_name_by_token: dict):
    """Return a function that turns a raw token into a readable label.

    icd_title_by_token: map "C:10_I509" → "Heart failure, unspecified"
                         (as stored in runs/full_CPD/icd_titles.json)
    drug_name_by_token:  map "D:RX_4094" → "Metolazone"
    """

    def short(tok: str, max_len: int = 45) -> str:
        if tok.startswith("D:"):
            name = drug_name_by_token.get(tok)
            if name:
                return f"D:{name}"
            return tok
        title = icd_title_by_token.get(tok, "")
        prefix = "C" if tok.startswith("C:") else ("P" if tok.startswith("P:") else tok[:1])
        code_part = tok.split("_", 1)[-1] if "_" in tok else tok
        if title:
            t = title if len(title) <= max_len else title[: max_len - 1] + "…"
            return f"{prefix}:{code_part} ({t})"
        return tok

    def short_code(tok: str) -> str:
        """Ultra-short label for pattern rendering — drug name or bare code only,
        no long ICD title. Used in Panel C so patterns fit on one line."""
        if tok.startswith("D:"):
            name = drug_name_by_token.get(tok)
            return f"D:{name}" if name else tok
        code = tok.split("_", 1)[-1] if "_" in tok else tok.split(":", 1)[-1]
        prefix = tok[:1]
        return f"{prefix}:{code}"

    def pattern(pat_str: str, max_tok_len: int = 60, terse: bool = False) -> str:
        # Episode patterns use " -> " between blocks; within block items are
        # comma-joined inside braces: "{X, Y} -> Z" or bare tokens "X -> Y -> Z".
        render_tok = short_code if terse else (lambda t: short(t, max_tok_len))

        def _render_block(block: str) -> str:
            s = block.strip()
            if s.startswith("{") and s.endswith("}"):
                inner = s[1:-1]
                parts = [p.strip() for p in inner.split(",")]
                return "{" + ", ".join(render_tok(p) for p in parts) + "}"
            return render_tok(s)

        blocks = [b.strip() for b in pat_str.split("->")]
        return " → ".join(_render_block(b) for b in blocks)

    return short, pattern

BUNDLE = "../runs/full_CPD"
MAIN_OUT = f"{BUNDLE}/explain/figures/main/fig2_patient_explanation_test.png"
SUBMIT_OUT = "../manuscript/jamia_submission/figures/Figure2.png"

# Exemplar we've been using all along
EXEMPLAR_PID = "16098031"
EXEMPLAR_HADM = "29583862"

RISK  = "#c0392b"
PROT  = "#2c7bb6"
SHAP_FACE = "#7f8c8d"   # SHAP bar face (grey-ish) — sign encoded by alpha edge
SHAP_EDGE = "#2c3e50"


# ── Load everything ────────────────────────────────────────────────────────
print("[1] Loading model, vocab, config...")
with open(f"{BUNDLE}/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vocab_inv = {idx: tok for tok, idx in vocab.items()}
with open(f"{BUNDLE}/config.json") as f:
    cfg = json.load(f)
with open(f"{BUNDLE}/icd_titles.json") as f:
    icd_title_by_token = json.load(f)   # keys like "C:10_I509" → title
# Drug name mapping (D:RX_<cui> → ingredient name)
drug_mapping_csv = "../data/mimiciv/3.1/hosp/ndc_to_ingredient.csv"
if os.path.exists(drug_mapping_csv):
    drug_name_by_token = load_drug_names(drug_mapping_csv)
else:
    drug_name_by_token = {}
fmt_tok, fmt_pat = make_token_formatter(icd_title_by_token, drug_name_by_token)
# Build a no-op icd_titles alias for format_token_readable calls that still rely on it
icd_titles = {}

state = torch.load(f"{BUNDLE}/model.pt", map_location="cpu", weights_only=False)
vocab_size = state["token_emb.weight"].shape[0]
embedding_dim = state["token_emb.weight"].shape[1]
max_seq_len = state["pos_emb.weight"].shape[0]
model = SingleStreamTransformer(
    vocab_size=vocab_size, embedding_dim=embedding_dim, max_seq_len=max_seq_len,
    num_heads=cfg.get("heads", 4), num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.1),
)
model.load_state_dict(state)
model.eval()
device = torch.device("cpu")

print("[2] Loading cached tensors, sequences, IG...")
tensors = torch.load(f"{BUNDLE}/ig_cache/tensors_test.pt", map_location="cpu", weights_only=False)
with open(f"{BUNDLE}/ig_cache/sequences_test.pkl", "rb") as f:
    df_seq_test = pickle.load(f)
df_ig_test = pd.read_csv(f"{BUNDLE}/ig_cache/ig_test.csv")

# Pick the exemplar
pick_rows = df_seq_test[
    (df_seq_test["patient_id"].astype(str) == EXEMPLAR_PID) &
    (df_seq_test["index_hadm_id"].astype(str) == EXEMPLAR_HADM)
]
assert len(pick_rows) == 1, f"Exemplar not found ({EXEMPLAR_PID}/{EXEMPLAR_HADM})"
pick = pick_rows.iloc[0].to_dict()
input_ids_tensor = tensors[(EXEMPLAR_PID, EXEMPLAR_HADM)]  # (1, 512)
print(f"    exemplar: pid={EXEMPLAR_PID}  hadm={EXEMPLAR_HADM}  "
      f"readmission={pick['readmission']}  n_visits={pick.get('n_visits', '?')}")

print("[3] Loading mined patterns...")
pat_files = sorted(glob.glob(f"{BUNDLE}/mining_cache/all_scored_*.csv"))
# Use the same file the paper uses (latest / largest)
df_patterns_train = pd.read_csv(pat_files[-1])
print(f"    mining file: {os.path.basename(pat_files[-1])}  rows={len(df_patterns_train):,}")


# ── Compute IG for the exemplar (fast — reuse cached ig_test.csv) ──────────
print("[4] Pulling cached IG values for this patient...")
patient_ig = df_ig_test[
    (df_ig_test["patient_id"].astype(str) == EXEMPLAR_PID) &
    (df_ig_test["index_hadm_id"].astype(str) == EXEMPLAR_HADM)
].copy()
assert len(patient_ig) > 0, "No IG rows for exemplar in cache"
# Build position-to-ig maps
tok_ig_signed = {}
for _, r in patient_ig.iterrows():
    tok_ig_signed[r["token_str"]] = float(r["ig_signed"])


# ── Build list of top-|IG| tokens (feature subset for SHAP) ────────────────
ids_np = input_ids_tensor[0].numpy()
items = []
for pos in range(len(ids_np)):
    tid = int(ids_np[pos])
    if tid == PAD_IDX:
        continue
    tok_str = vocab_inv.get(tid, f"UNK_{tid}")
    if tok_str in SPECIAL_TOKENS:
        continue
    ig_signed = tok_ig_signed.get(tok_str)
    # prefer cached ig from the CSV; fall back to 0 if token repeats across positions
    items.append({
        "pos": pos,
        "token_str": tok_str,
        "token_str_readable": fmt_tok(tok_str),
        "ig_signed": float(ig_signed) if ig_signed is not None else 0.0,
        "ig_abs": abs(float(ig_signed)) if ig_signed is not None else 0.0,
    })
items.sort(key=lambda d: d["ig_abs"], reverse=True)
SHAP_K = cfg.get("shap_max_features", 20)
items_sel = items[:SHAP_K]
print(f"    top-|IG| tokens for SHAP: K={len(items_sel)}")


# ── SHAP (masking-based) — fast for 1 patient ──────────────────────────────
import shap
print(f"[5] Running KernelSHAP ({SHAP_K} features, {cfg.get('shap_nsamples', 200)} samples)...")
base_ids = input_ids_tensor.clone()


def predict_masked(z):
    batch = []
    for row in z:
        inp = base_ids.clone()
        for j, keep in enumerate(row.tolist()):
            if keep < 0.5:
                inp[0, items_sel[j]["pos"]] = PAD_IDX
        batch.append(inp)
    stacked = torch.cat(batch, dim=0)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(stacked), 32):
            chunk = stacked[i:i+32].to(device)
            out = model(chunk)
            all_probs.append(out["y_prob"].cpu().numpy())
    return np.concatenate(all_probs, axis=0)


rng_bg = np.random.RandomState(42)
background = rng_bg.randint(0, 2, size=(10, SHAP_K)).astype(float)
x_instance = np.ones((1, SHAP_K))
explainer = shap.KernelExplainer(predict_masked, background)
shap_vals = explainer.shap_values(x_instance, nsamples=cfg.get("shap_nsamples", 200))
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
shap_vals = np.array(shap_vals).reshape(-1)
shap_for_tok = {items_sel[i]["token_str"]: float(shap_vals[i]) for i in range(len(items_sel))}
print(f"    |SHAP| max = {np.max(np.abs(shap_vals)):.4f}")


# ── Match patterns for this patient ─────────────────────────────────────────
print("[6] Matching patterns against patient's salient visit blocks...")
salient_visit_blocks = pick.get("salient_visit_blocks", [])
mining_method = cfg.get("mining_method", "episode")
is_episode = (mining_method == "episode")
all_matched = _match_all_patterns(df_patterns_train, salient_visit_blocks, is_episode)
matched_scored = []
for pat_str, orr, sup, pat_tokens in all_matched:
    ig_sum = sum(tok_ig_signed.get(t, 0.0) for t in pat_tokens)
    matched_scored.append({
        "pattern": pat_str,
        "odds_ratio": float(orr),
        "support": int(sup),
        "tokens": pat_tokens,
        "ig_signed_sum": ig_sum,
    })
print(f"    matched {len(matched_scored)} patterns")


# ──────────────────────────────────────────────────────────────────────────
#  Build the new 3-panel figure
# ──────────────────────────────────────────────────────────────────────────
print("[7] Rendering figure...")

fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(
    nrows=2, ncols=10,
    height_ratios=[1.0, 1.20],
    hspace=0.34, wspace=1.8,       # plenty of horizontal breathing room
    left=0.05, right=0.99, top=0.93, bottom=0.05,
)
axA = fig.add_subplot(gs[0, :])        # top — full width
axB = fig.add_subplot(gs[1, 0:4])      # bottom-left — visit timeline
axC = fig.add_subplot(gs[1, 4:10])     # bottom-right — patterns (wider for long labels)

fig.suptitle(
    f"Representative {cfg.get('test_split_name', 'test')} patient — "
    f"{pick.get('n_visits', '?')} visits, readmitted, "
    f"{len(matched_scored)} matched patterns",
    fontsize=14, fontweight="bold", y=0.98)


# ── Panel A: SHAP vs IG agreement (paired bars, normalized) ────────────────
shap_top = sorted(items_sel, key=lambda d: abs(shap_for_tok[d["token_str"]]), reverse=True)[:8]
ig_top = sorted(items_sel, key=lambda d: d["ig_abs"], reverse=True)[:8]
union_tokens = []
seen = set()
for it in shap_top + ig_top:
    if it["token_str"] not in seen:
        union_tokens.append(it["token_str"])
        seen.add(it["token_str"])
# Rank union by |IG_signed| descending for display ordering (most impactful at top)
union_tokens = sorted(union_tokens, key=lambda t: -abs(tok_ig_signed.get(t, 0.0)))[:12]

# Normalize each method to [-1, 1] so bars share scale
shap_vec = np.array([shap_for_tok[t] for t in union_tokens])
ig_vec   = np.array([tok_ig_signed[t] for t in union_tokens])
shap_norm = shap_vec / (np.max(np.abs(shap_vec)) + 1e-12)
ig_norm   = ig_vec   / (np.max(np.abs(ig_vec))   + 1e-12)

# y ordering: largest |IG| at top — reverse for matplotlib's bottom-up axis
y = np.arange(len(union_tokens))[::-1]
token_labels = [fmt_tok(t) for t in union_tokens]

bar_h = 0.36
# IG bars (primary) — colored red/blue by sign
ig_colors = [RISK if v > 0 else PROT for v in ig_norm]
bars_ig = axA.barh(y - bar_h / 2, ig_norm, height=bar_h,
                   color=ig_colors, edgecolor="black", linewidth=0.4,
                   label="Integrated Gradients (normalized)")
# SHAP bars (secondary) — same sign-based color but lighter + hatched
shap_colors_light = [
    ("#e78a7d" if v > 0 else "#90c4e3") for v in shap_norm
]
bars_shap = axA.barh(y + bar_h / 2, shap_norm, height=bar_h,
                     color=shap_colors_light, edgecolor="black", linewidth=0.4,
                     hatch="///", label="SHAP (normalized)")

axA.set_yticks(y)
axA.set_yticklabels(token_labels, fontsize=11)
axA.axvline(0, color="black", linewidth=0.8)
axA.set_xlim(-1.1, 1.1)
axA.set_xlabel(
    "Normalized attribution (each method scaled by its own max |value|)\n"
    "← protective          risk →",
    fontsize=11,
)
axA.set_title(
    "A · Token-level attribution agreement: SHAP vs Integrated Gradients\n"
    f"(Union of top-8 tokens by each method, n={len(union_tokens)}; "
    f"bars aligned means methods agree; divergence = method-specific signal)",
    fontsize=12, loc="left", pad=8,
)

# Legend: color = risk/protective, texture = method
legend_elems = [
    Patch(facecolor=RISK, edgecolor="black", linewidth=0.4, label="Risk (attribution > 0)"),
    Patch(facecolor=PROT, edgecolor="black", linewidth=0.4, label="Protective (attribution < 0)"),
    Patch(facecolor="white", edgecolor="black", linewidth=0.4, label="IG (solid)"),
    Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=0.4, label="SHAP (hatched)"),
]
axA.legend(handles=legend_elems, loc="lower right", fontsize=10, framealpha=0.95, ncol=2)
for side in ("top", "right"):
    axA.spines[side].set_visible(False)


# ── Panel B: Visit-level attribution timeline ──────────────────────────────
# One column per visit; tokens within a visit are stacked vertically as bars.
# Bar length = |IG|, color = sign. Cumulative signed IG line overlaid.
axB.set_title(
    "B · Visit-level attribution timeline\n"
    f"({len(salient_visit_blocks)} visits; bars sized by |IG|, "
    "colored by direction; dashed line = cumulative signed IG)",
    fontsize=12, loc="left", pad=8,
)

n_visits = len(salient_visit_blocks)
# Compute per-visit net signed IG (sum of all tokens in that visit)
visit_net = []
visit_top_tok = []     # (token_str, signed_ig) of the strongest contributor
for block in salient_visit_blocks:
    s = 0.0
    best_tok, best_mag = None, 0.0
    for tok in block:
        v = tok_ig_signed.get(tok, 0.0)
        s += v
        if abs(v) > abs(best_mag):
            best_tok, best_mag = tok, v
    visit_net.append(s)
    visit_top_tok.append((best_tok, best_mag))

# Bar plot: one bar per visit, height = net signed IG.
visit_colors = [RISK if v > 0 else PROT for v in visit_net]
bars = axB.bar(range(n_visits), visit_net, width=0.65,
               color=visit_colors, edgecolor="black", linewidth=0.4)

# Annotate each bar with the dominant token of that visit (above for risk, below for protective).
ymax = max((abs(v) for v in visit_net), default=1e-6)
for i, (net_v, (tok, _)) in enumerate(zip(visit_net, visit_top_tok)):
    if tok is None:
        continue
    if tok.startswith("D:"):
        label = fmt_tok(tok)
    else:
        code = tok.split("_", 1)[-1] if "_" in tok else tok.split(":", 1)[-1]
        label = f"{tok[:1]}:{code}"
    if len(label) > 22:
        label = label[:20] + "…"
    dy = ymax * 0.04
    y_text = net_v + (dy if net_v >= 0 else -dy)
    va = "bottom" if net_v >= 0 else "top"
    axB.text(i, y_text, label, ha="center", va=va, fontsize=9, color="#333333")

# Cumulative signed IG line overlaid on secondary axis
cum = np.cumsum(visit_net)
axB_r = axB.twinx()
axB_r.plot(range(n_visits), cum, color="#555555", linestyle="--",
           linewidth=1.7, marker="o", markersize=6, zorder=3)
axB_r.set_ylabel("")   # left blank — legend already labels the dashed line
axB_r.tick_params(axis="y", colors="#555555", labelsize=9)
axB_r.spines["top"].set_visible(False)
axB_r.spines["right"].set_color("#999999")

axB.set_xticks(range(n_visits))
axB.set_xticklabels([f"V{i+1}" for i in range(n_visits)], fontsize=11)
axB.set_xlim(-0.6, n_visits - 0.4)
axB.set_ylabel("Visit net signed IG", fontsize=11)
axB.axhline(0, color="black", linewidth=0.6)
# Give the annotations a little vertical room
axB.set_ylim(-ymax * 1.35, ymax * 1.35)
for side in ("top", "right"):
    axB.spines[side].set_visible(False)
axB.set_xlabel("Visit (chronological)", fontsize=11)

leg_b = [
    Patch(facecolor=RISK, label="Risk visit (net IG > 0)"),
    Patch(facecolor=PROT, label="Protective visit (net IG < 0)"),
    plt.Line2D([0], [0], color="#555555", linestyle="--", marker="o",
               label="Cumulative signed IG (right axis)"),
]
axB.legend(handles=leg_b, loc="upper left", fontsize=9, framealpha=0.95)


# ── Panel C: Top-6 matched patterns with FULL names ────────────────────────
N_TOP_PAT = 6
sorted_matched = sorted(matched_scored, key=lambda m: abs(m["ig_signed_sum"]), reverse=True)
top_patterns = sorted_matched[:N_TOP_PAT]
# Reverse for display so most impactful is at top
top_patterns = top_patterns[::-1]

axC.set_title(
    f"C · Top {N_TOP_PAT} matched temporal patterns\n"
    f"(of {len(matched_scored)} total; ranked by |Σ IG|; full names, OR + carrier count annotated)",
    fontsize=12, loc="left", pad=8,
)

if top_patterns:
    vals = [p["ig_signed_sum"] for p in top_patterns]
    colors = [RISK if v > 0 else PROT for v in vals]
    # Hard cap prevents long drug names from overflowing Panel B
    MAX_LABEL = 32
    raw_labels = [fmt_pat(p["pattern"], terse=True) for p in top_patterns]
    labels = [(lab if len(lab) <= MAX_LABEL else lab[:MAX_LABEL - 1] + "…")
              for lab in raw_labels]
    anno   = [f"OR {p['odds_ratio']:.2f} · n={p['support']}" for p in top_patterns]

    y_pos = np.arange(len(top_patterns))
    axC.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.4)
    axC.set_yticks(y_pos)
    axC.set_yticklabels(labels, fontsize=10.5)
    axC.axvline(0, color="black", linewidth=0.8)
    axC.set_xlabel("Σ IG over pattern tokens\n← protective          risk →", fontsize=11)

    # Annotation drawn INSIDE the bar (white text on colored fill) so nothing
    # spills outside the axes. Place at 90% of the bar's end.
    xmax = max(abs(v) for v in vals) if vals else 1.0
    for i, (v, a) in enumerate(zip(vals, anno)):
        inside_x = v * 0.90
        axC.text(inside_x, i, a,
                 ha=("right" if v > 0 else "left"),
                 va="center", fontsize=9.5, color="white", fontweight="bold")
    # Small headroom so axis ticks don't crowd the bars
    axC.set_xlim(-xmax * 1.10, xmax * 1.10)
    for side in ("top", "right"):
        axC.spines[side].set_visible(False)
else:
    axC.text(0.5, 0.5, "No patterns matched this patient.", ha="center", va="center", fontsize=11)


# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MAIN_OUT), exist_ok=True)
fig.savefig(MAIN_OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[8] Saved: {MAIN_OUT}")

# Copy to submission folder
import shutil
os.makedirs(os.path.dirname(SUBMIT_OUT), exist_ok=True)
shutil.copyfile(MAIN_OUT, SUBMIT_OUT)
print(f"    copied → {SUBMIT_OUT}")

# Also save the backing CSV for reproducibility (union-top tokens + matched patterns)
csv_dir = f"{BUNDLE}/explain/figures/csv"
os.makedirs(csv_dir, exist_ok=True)
union_df = pd.DataFrame([
    {
        "token_str": t,
        "token_readable": fmt_tok(t),
        "shap": shap_for_tok.get(t, 0.0),
        "ig_signed": tok_ig_signed.get(t, 0.0),
    }
    for t in union_tokens
])
union_df.to_csv(f"{csv_dir}/fig2_top_tokens_union_test.csv", index=False)
pd.DataFrame(matched_scored).to_csv(f"{csv_dir}/fig2_matched_patterns_test.csv", index=False)
print(f"    CSVs saved in {csv_dir}/")
