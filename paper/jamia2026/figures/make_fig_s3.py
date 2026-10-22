"""make_fig_s3, Supplementary Figure S3: SHAP vs IG agreement.

Two-panel scatter:
  S3a, patient-level scatter for the main_tp exemplar's top-K tokens
  S3b, aggregate scatter pooling top-K tokens from N sampled test patients

Uses per-patient KernelSHAP on the top-K-by-IG token subset (not the
cohort SHAP cache, which is computed on the full all-tokens space at low
nsamples and is suitable for token ranking but not per-patient magnitude
scatter, see docstring of build_cohort_shap_cache.py).

KernelSHAP runs the model many times per patient. Defaults are tuned for a
~15 min CPU run; bump ``--sample-n`` if the aggregate panel reads
underpopulated. Use a GPU host for substantially faster runs.

Usage:
    python paper/jamia2026/figures/make_fig_s3.py --run_dir ../runs/mimiciv_readmission_30d --device mps \
            --shap-nsamples 1000
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from atop.config import PAD_IDX
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.run_layout import RunLayout
from atop.utils import load_drug_names


RISK = "#c44e52"   # canonical risk red (matches PHEN_COLORS / Fig S1)
PROT = "#4c72b0"   # canonical protective blue (matches PHEN_COLORS / Fig S1)


def _fmt_token(tok: str, drug_names: dict) -> str:
    if tok.startswith("D:"):
        n = drug_names.get(tok)
        return f"D:{n}" if n else tok
    code = tok.split("_", 1)[-1] if "_" in tok else tok.split(":", 1)[-1]
    return f"{tok[:1]}:{code}"


def _shap_for_patient(
    pid: str, hadm: str, top_k: int,
    *, tensors, ig_all, model, device, shap_nsamples: int, chunk: int, seed: int,
):
    """Return ``{token_str: (ig_signed, shap_val)}`` for the top-K tokens
    by ``|IG|``, or ``None`` if the patient lacks a tensor / IG row.

    Pure compute, no caching here. Use :func:`compute_with_cache` for the
    cached version called from the main loop.
    """
    import shap as shap_lib

    key = (pid, hadm)
    if key not in tensors:
        return None
    input_ids_tensor = tensors[key]
    rows = ig_all[(ig_all["patient_id"] == pid) & (ig_all["index_hadm_id"] == hadm)]
    if len(rows) == 0:
        return None
    rows = rows.sort_values("ig_abs", ascending=False).head(top_k).reset_index(drop=True)
    feats = [(int(r["pos"]), r["token_str"], float(r["ig_signed"]))
             for _, r in rows.iterrows()]
    K = len(feats)
    if K == 0:
        return None

    base_ids = input_ids_tensor.clone()

    def predict_masked(z):
        batch = []
        for row in z:
            inp = base_ids.clone()
            for j, keep in enumerate(row.tolist()):
                if keep < 0.5:
                    inp[0, feats[j][0]] = PAD_IDX
            batch.append(inp)
        stacked = torch.cat(batch, dim=0)
        all_p = []
        with torch.no_grad():
            for i in range(0, len(stacked), chunk):
                ch = stacked[i:i + chunk].to(device)
                out = model(ch)
                all_p.append(out["y_prob"].cpu().numpy())
        return np.concatenate(all_p, axis=0)

    rng_bg = np.random.RandomState(seed)
    bg = rng_bg.randint(0, 2, size=(10, K)).astype(float)
    x_inst = np.ones((1, K))
    expl = shap_lib.KernelExplainer(predict_masked, bg)
    sv = expl.shap_values(x_inst, nsamples=shap_nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).reshape(-1)
    return {feats[i][1]: (feats[i][2], float(sv[i])) for i in range(K)}


# ════════════════════════════════════════════════════════════════════════
# SHAP cache, single-file dict at <run_dir>/3_attribution/shap_cache.pkl
# Keyed by (pid, hadm, top_k, shap_nsamples, seed) so different parameter
# choices coexist. Saved every ``flush_every`` new entries to survive
# interruptions; one full GPU run populates the cache and subsequent local
# runs are instant.
# ════════════════════════════════════════════════════════════════════════


def load_shap_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  [shap-cache] loaded {len(cache)} entries from {cache_path}")
        return cache
    print(f"  [shap-cache] no existing cache at {cache_path} (first run)")
    return {}


def save_shap_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)


def compute_with_cache(
    pid: str, hadm: str, top_k: int,
    *, tensors, ig_all, model, device, shap_nsamples: int, chunk: int, seed: int,
    cache: dict, cache_path: str, flush_every: int = 5,
):
    """Cached wrapper around :func:`_shap_for_patient`. Returns the same
    ``{token_str: (ig_signed, shap_val)}`` dict (or ``None``)."""
    key = (str(pid), str(hadm), int(top_k), int(shap_nsamples), int(seed))
    if key in cache:
        return cache[key], True  # (result, hit)

    result = _shap_for_patient(
        pid, hadm, top_k,
        tensors=tensors, ig_all=ig_all, model=model, device=device,
        shap_nsamples=shap_nsamples, chunk=chunk, seed=seed,
    )
    if result is not None:
        cache[key] = result
        # Periodic flush so a ctrl-C halfway through doesn't lose work
        if len(cache) % flush_every == 0:
            save_shap_cache(cache, cache_path)
    return result, False


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S3 (SHAP vs IG)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--sample-n", type=int, default=100,
                   help="Number of test patients for the aggregate panel")
    p.add_argument("--top-k-tokens", type=int, default=20,
                   help="SHAP features per patient for Panel S3a")
    p.add_argument("--agg-top-k", type=int, default=10,
                   help="Tokens per patient pooled in Panel S3b")
    p.add_argument("--shap-nsamples", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS3_shap_vs_ig.png")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory for output CSVs (default: <run_dir>/5_analysis)")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip the SHAP cache and recompute every patient")
    p.add_argument("--cache-flush-every", type=int, default=5,
                   help="Save the cache after this many new entries (interruption-safe)")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir

    # ── Load model + caches ───────────────────────────────────────────
    print("[1] Loading model + vocab + cached IG + tensors...", flush=True)
    with open(layout.vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(layout.config_json) as f:
        cfg = json.load(f)
    drug_names = load_drug_names(os.path.join(args.mimic_dir, "ndc_to_ingredient.csv"))

    state = torch.load(layout.model_pt, map_location="cpu", weights_only=False)
    vocab_size = state["token_emb.weight"].shape[0]
    emb_dim = state["token_emb.weight"].shape[1]
    max_seq_len = state["pos_emb.weight"].shape[0]
    model = SingleStreamTransformer(
        vocab_size=vocab_size, embedding_dim=emb_dim, max_seq_len=max_seq_len,
        num_heads=cfg.get("heads", 4), num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    tensors = torch.load(layout.tensors_pt("test"),
                         map_location="cpu", weights_only=False)
    ig_all = pd.read_csv(
        layout.ig_csv("test"),
        dtype={"patient_id": str, "index_hadm_id": str},
    )

    with open(layout.exemplars_json) as f:
        exemplar = json.load(f)["main_tp"]
    EX_PID, EX_HADM = exemplar["patient_id"], exemplar["index_hadm_id"]

    # ── SHAP cache ────────────────────────────────────────────────────
    cache_path = layout.shap_cache_pkl
    cache: dict = {} if args.no_cache else load_shap_cache(cache_path)

    # ── Panel S3a: patient-level ──────────────────────────────────────
    print(f"[2] SHAP for main_tp pid={EX_PID} (top-{args.top_k_tokens} tokens, "
          f"nsamples={args.shap_nsamples})...", flush=True)
    pts, hit = compute_with_cache(
        EX_PID, EX_HADM, args.top_k_tokens,
        tensors=tensors, ig_all=ig_all, model=model, device=device,
        shap_nsamples=args.shap_nsamples, chunk=args.chunk, seed=args.seed,
        cache=cache, cache_path=cache_path, flush_every=args.cache_flush_every,
    )
    print(f"    {'cache hit' if hit else 'computed fresh'}", flush=True)
    pa_df = pd.DataFrame([
        {"token_str": t, "token_readable": _fmt_token(t, drug_names),
         "ig_signed": ig_s, "shap_signed": shap_s}
        for t, (ig_s, shap_s) in pts.items()
    ])
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(pa_df["ig_signed"], pa_df["shap_signed"])
    rho_p, _ = spearmanr(pa_df["ig_signed"], pa_df["shap_signed"])
    print(f"    Patient-level Pearson r = {r_p:.3f}  Spearman ρ = {rho_p:.3f}", flush=True)

    # ── Panel S3b: aggregate ──────────────────────────────────────────
    print(f"[3] Aggregate scatter: sampling {args.sample_n} test patients (nsamples={args.shap_nsamples})...", flush=True)
    candidates = ig_all[["patient_id", "index_hadm_id"]].drop_duplicates()
    candidates = candidates[
        (candidates["patient_id"] != EX_PID)
        | (candidates["index_hadm_id"] != EX_HADM)
    ]
    sample = candidates.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)

    agg_rows, n_done, n_hits, n_miss = [], 0, 0, 0
    for _, r in sample.iterrows():
        pid, hadm = r["patient_id"], r["index_hadm_id"]
        try:
            res, hit = compute_with_cache(
                pid, hadm, args.agg_top_k,
                tensors=tensors, ig_all=ig_all, model=model, device=device,
                shap_nsamples=args.shap_nsamples, chunk=args.chunk, seed=args.seed,
                cache=cache, cache_path=cache_path, flush_every=args.cache_flush_every,
            )
        except Exception:
            continue
        if res is None:
            continue
        if hit:
            n_hits += 1
        else:
            n_miss += 1
        for tok, (ig_s, shap_s) in res.items():
            agg_rows.append({"patient_id": pid, "index_hadm_id": hadm,
                             "token_str": tok, "ig_signed": ig_s, "shap_signed": shap_s})
        n_done += 1
        if n_done % 10 == 0:
            print(f"    progress: {n_done}/{len(sample)}  "
                  f"(cache: {n_hits} hit, {n_miss} computed)", flush=True)

    # Final flush so the last few entries are durable
    if not args.no_cache and n_miss > 0:
        save_shap_cache(cache, cache_path)
        print(f"  [shap-cache] final flush → {len(cache)} entries at {cache_path}", flush=True)

    agg_df = pd.DataFrame(agg_rows)
    print(f"    aggregate points: {len(agg_df):,}  "
          f"({n_hits} cached, {n_miss} freshly computed)", flush=True)
    if len(agg_df):
        r_a, _ = pearsonr(agg_df["ig_signed"], agg_df["shap_signed"])
        rho_a, _ = spearmanr(agg_df["ig_signed"], agg_df["shap_signed"])
        print(f"    Aggregate Pearson r = {r_a:.3f}  Spearman ρ = {rho_a:.3f}", flush=True)
    else:
        r_a = rho_a = float("nan")

    # ── Render ────────────────────────────────────────────────────────
    print("[4] Rendering figure...", flush=True)

    # Compute per-patient rank-overlap@K (additional confidence metric).
    # For Panel B aggregate: pool top-K IG vs top-K SHAP per patient,
    # then measure the per-patient Jaccard at K = min(5, agg_top_k).
    K_OVERLAP = min(5, args.agg_top_k)
    if len(agg_df):
        jaccards = []
        for (pid, hadm), g in agg_df.groupby(["patient_id", "index_hadm_id"]):
            top_ig   = set(g.nlargest(K_OVERLAP, key=lambda r: 0,
                                       )["token_str"]) if False else set(
                g.assign(abs_ig=g["ig_signed"].abs())
                 .nlargest(K_OVERLAP, "abs_ig")["token_str"])
            top_shap = set(
                g.assign(abs_shap=g["shap_signed"].abs())
                 .nlargest(K_OVERLAP, "abs_shap")["token_str"])
            inter = len(top_ig & top_shap)
            union = len(top_ig | top_shap)
            if union > 0:
                jaccards.append(inter / union)
        jacc_mean = float(np.mean(jaccards)) if jaccards else float("nan")
        jacc_med  = float(np.median(jaccards)) if jaccards else float("nan")
    else:
        jacc_mean = jacc_med = float("nan")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.5),
                                    gridspec_kw={"wspace": 0.30})

    # ════ Panel S3a, patient-level ════
    colors_a = [RISK if v > 0 else PROT for v in pa_df["ig_signed"]]
    axA.scatter(pa_df["ig_signed"], pa_df["shap_signed"],
                s=85, c=colors_a, edgecolor="black", linewidth=0.5, alpha=0.90,
                zorder=3)
    # IG is on the logit scale and KernelSHAP on the probability scale, so
    # the two axes are scaled independently; a shared y = x line would
    # flatten every SHAP value onto zero (reviewer issue #23).
    lim_x = pa_df["ig_signed"].abs().max() * 1.20
    lim_y = pa_df["shap_signed"].abs().max() * 1.20
    axA.axhline(0, color="#888888", linewidth=0.6, zorder=1)
    axA.axvline(0, color="#888888", linewidth=0.6, zorder=1)
    axA.set_xlim(-lim_x, lim_x)
    axA.set_ylim(-lim_y, lim_y)

    # Label only the top 6 tokens by |IG| (the most extreme dots). Skip
    # the cluster near origin to avoid the overlapping-label pile-up.
    N_LABEL = 6
    pa_labeled = pa_df.assign(abs_ig=pa_df["ig_signed"].abs()) \
                       .nlargest(N_LABEL, "abs_ig")
    for _, row in pa_labeled.iterrows():
        label = _fmt_token(row["token_str"], drug_names)
        if len(label) > 22:
            label = label[:19] + "..."
        # Offset away from the diagonal so labels don't sit on top of dots
        dx = 6 if row["ig_signed"] >= 0 else -6
        dy = 6 if row["shap_signed"] >= 0 else -6
        ha = "left" if dx > 0 else "right"
        axA.annotate(label,
                     xy=(row["ig_signed"], row["shap_signed"]),
                     xytext=(dx, dy), textcoords="offset points",
                     fontsize=9, ha=ha, va="center",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                ec="#bbbbbb", lw=0.5, alpha=0.92),
                     zorder=4)

    axA.set_xlabel("Integrated Gradients (signed)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axA.set_ylabel("KernelSHAP (signed)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axA.set_title(
        f"A · Exemplar patient: top {args.top_k_tokens} tokens by |IG|\n"
        f"Pearson r = {r_p:.3f}    Spearman ρ = {rho_p:.3f}",
        fontsize=12, loc="left", pad=10,
    )
    axA.tick_params(labelsize=10)
    axA.grid(True, color="#eeeeee", linewidth=0.5)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)
    axA.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ════ Panel S3b, aggregate (z-scored for scale comparability) ════
    if len(agg_df):
        # Z-score each metric so the comparable shape (not the absolute
        # magnitude) is visible. IG and SHAP have intrinsically different
        # scales (different normalizations); the rank/sign agreement is
        # what we want to show, not the unit-for-unit match.
        ig_z = (agg_df["ig_signed"] - agg_df["ig_signed"].mean()) \
               / max(agg_df["ig_signed"].std(), 1e-9)
        shap_z = (agg_df["shap_signed"] - agg_df["shap_signed"].mean()) \
                 / max(agg_df["shap_signed"].std(), 1e-9)
        colors_b = [RISK if v > 0 else PROT for v in agg_df["ig_signed"]]
        axB.scatter(ig_z, shap_z,
                    s=14, c=colors_b, edgecolor="none", alpha=0.40,
                    zorder=3)
        lim_b = max(ig_z.abs().max(), shap_z.abs().max()) * 1.05
        axB.plot([-lim_b, lim_b], [-lim_b, lim_b],
                 color="#888888", linestyle="--", linewidth=1.0,
                 label="y = x (z-scored)", zorder=1)
        axB.axhline(0, color="#888888", linewidth=0.6, zorder=1)
        axB.axvline(0, color="#888888", linewidth=0.6, zorder=1)
        axB.set_xlim(-lim_b, lim_b)
        axB.set_ylim(-lim_b, lim_b)
    else:
        axB.text(0.5, 0.5, "No aggregate points collected.",
                 transform=axB.transAxes, ha="center", va="center")

    axB.set_xlabel("Integrated Gradients (z-score)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axB.set_ylabel("KernelSHAP (z-score)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axB.set_title(
        f"B · Aggregate: {n_done} test patients × top-{args.agg_top_k}\n"
        f"n = {len(agg_df):,} pairs    Pearson r = {r_a:.3f}    "
        f"Spearman ρ = {rho_a:.3f}    "
        f"Per-patient Jaccard@{K_OVERLAP} = {jacc_mean:.2f} (mean)",
        fontsize=12, loc="left", pad=10,
    )
    axB.tick_params(labelsize=10)
    axB.grid(True, color="#eeeeee", linewidth=0.5)
    for side in ("top", "right"):
        axB.spines[side].set_visible(False)
    axB.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # No in-image suptitle; the DOCX caption supplies "Figure S3. ...".
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[5] Saved figure: {args.output_path} ({size_kb} KB)", flush=True)

    os.makedirs(csv_dir, exist_ok=True)
    pa_df.to_csv(os.path.join(csv_dir, "fig_s3_patient_level.csv"), index=False)
    agg_df.to_csv(os.path.join(csv_dir, "fig_s3_aggregate.csv"), index=False)
    print(f"    csvs in {csv_dir}/", flush=True)


if __name__ == "__main__":
    main()
