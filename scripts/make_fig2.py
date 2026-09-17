"""make_fig2, Figure 2 (single-patient AToP explanation, v2 redesign).

Two-panel layout:

  TOP, Visit timeline with pattern unlocks at the index visit.
        Visits as boxes left-to-right with day-offset labels.
        Index visit (final visit in the model's input sequence) highlighted
        with heavy border + "INDEX VISIT" label. The R30 outcome marker
        sits to the right of the index visit (indicating what happened in
        the 30-day window AFTER discharge from the index admission).

        All matched patterns "unlock" at the index visit (the prediction
        point integrates everything available by then). Each pattern is
        rendered with arrows/markers tying it to the visit(s) that supplied
        its constituent tokens:
          - Cross-visit (A -> B):     curved arc from earlier visit to V_idx
          - Single-visit cooc ({A;B}): bracket marker within one visit
          - Single-token (A):         dot at the visit hosting the token

  BOTTOM, Greedy non-overlapping pattern waterfall.
        Sort matched patterns by |sigma_IG| descending. Walk down: each
        pattern claims previously-unclaimed (position, token) pairs in the
        patient's salient set. The pattern's net contribution = sum of IG
        over newly-claimed positions. Patterns whose required positions are
        entirely claimed by higher-impact predecessors are dropped.

        Result: every salient position appears in exactly one bar; sum of
        bars = patient's salient sigma_IG (no over-attribution). Glyphs in
        the y-tick labels distinguish pattern type.

Why "unlock at index visit": the model makes a single binary prediction at
the end of the input sequence, "will this patient be readmitted within 30
days of discharge from the most recent admission?", and that prediction
integrates ALL pattern evidence available up to and including the index
visit. So from a clinician's perspective, the index visit IS the moment of
prediction; patterns that completed earlier are already part of the input.
The arrows show provenance, not temporal unlock.

Usage:
    python scripts/make_fig2.py --run_dir ../runs/full_CPD_v2
    python scripts/make_fig2.py --run_dir ../runs/full_CPD_v2 \
                                --pid 11506385 --hadm 21463919

Output: <output_path>  (default: ../manuscript/jamia_submission/figures/Figure2.png,
        auto-suffixed with _<pid> unless --no-id-suffix is set)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Set

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.explain.matching import _match_all_patterns
from atop.explain.pattern_labels import PatternLabelResolver
from atop.mining.patterns import _parse_episode_pattern, _match_subpattern
from atop.run_layout import RunLayout


# ── Visual constants ──────────────────────────────────────────────────────
RISK_COLOR     = "#ca0020"
PROT_COLOR     = "#2166ac"
NEUTRAL_COLOR  = "#888888"
# Readmission marker: a distinct darker red so the OUTCOME visually
# separates from the risk-pushing token text (which uses RISK_COLOR).
# Same hue family, more saturated/darker, reads as "this is the event"
# rather than blending with the token IG annotations.
# Readmission outcome reuses the same red/blue as the per-token /
# per-pattern attribution colors. The outcome is differentiated by
# SHAPE (diamond) and WORDS ("Patient readmitted"), not by hue,
# keeps the figure to a single red and a single blue end-to-end.
R30_RISK_COLOR    = RISK_COLOR
R30_PROT_COLOR    = PROT_COLOR
TAIL_COLOR        = "#9aa0a6"   # medium gray, Panel A "tail" tokens
                                # (high |IG| but NOT surfaced in Panel B's
                                # top-N patterns). De-emphasized so the
                                # eye focuses on surfaced (colored) tokens.
INDEX_FACECOL  = "#fff7e6"
INDEX_EDGECOL  = "#000000"
VISIT_FACECOL  = "#f4f4f4"
VISIT_EDGECOL  = "#888888"
ARC_ALPHA      = 0.55

# Pattern-type glyphs (deprecated, kept for reference; v8+ uses visit-reach
# prefix in labels instead, e.g. "V1→V5: ...")
GLYPH_CROSS_VISIT = "═"
GLYPH_COOCCURRENCE = "●"
GLYPH_SINGLE_TOKEN = "○"
GLYPH_RESIDUAL     = "░"


# ── Label abbreviations for waterfall display ────────────────────────────
# Strategy mirrors Figure 3: short token names live in scripts/short_names.csv
# and are loaded via PatternLabelResolver. We only post-process here for the
# rare patient's noncompliance phrase (it's an event in MIMIC titles, not a
# token code, so it can't be added to short_names.csv as a code mapping).
# Per-token-code overrides: when distinct ICD-10 codes resolve to the
# same generic English phrase (e.g., "Secondary malig. neoplasm"),
# carry the anatomy explicitly so the figure doesn't show two visits
# with what looks like the same finding. Keyed by full token_str.
TOKEN_LABEL_OVERRIDES = {
    # Secondary / unspecified malignant neoplasm, multiple codes, each
    # a distinct anatomic site. Without these overrides, all three
    # render as "Secondary malig. neoplasm" (visually identical).
    "C:10_C786": "Secondary malig. (retroperitoneal)",                # retroperit + peritoneum
    "C:10_C771": "Secondary malig. (intrathoracic lymph nodes)",      # intrathoracic LN
    "C:10_C775": "Secondary malig. (intrapelvic lymph nodes)",        # intrapelvic LN
    "C:10_C774": "Secondary malig. (inguinal lymph nodes)",           # inguinal LN
    "C:10_C782": "Secondary malig. (pleural)",                        # pleura
}


LABEL_ABBREV = [
    ("Patient's noncompliance with dietary regimen", "Dietary noncompliance"),
    ("Patient's noncompliance with dietary r...", "Dietary noncompliance"),
    ("Anemia due to antineoplastic chemotherapy", "Chemo-induced anemia"),
    # ── Long oncology / surgical names that overflow visit boxes ──
    ("Resection of right fallopian tube", "Resect. R fallopian tube"),
    ("Resection of right fallopian", "Resect. R fallopian tube"),
    ("Resection of Cervix, Open Approach", "Resect. cervix (open)"),
    ("Resection of cervix, open approach", "Resect. cervix (open)"),
    ("Family Hx malignant neoplasm", "Fam Hx malig. neoplasm"),
    ("Family history of malignant neoplasm", "Fam Hx malig. neoplasm"),
    # Generic catch-all for Secondary malig. neoplasm, applies when
    # the per-code anatomy dispatch (TOKEN_CODE_ANATOMY below, used by
    # fmt_token_anatomy in main()) doesn't match. Code-level dispatch
    # is the primary mechanism for keeping distinct ICD-10 codes
    # visually distinct; this string-level rule is a fallback for
    # rendering paths that don't go through fmt_token_anatomy.
    ("Secondary malignant neoplasm of unspecified site",
     "Secondary malig. neoplasm"),
    ("Secondary malignant neoplasm", "Secondary malig. neoplasm"),
    ("Drainage of peritoneal cavity (Re...)", "Peritoneal drainage"),
    ("Drainage of peritoneal cavity", "Peritoneal drainage"),
    ("Retention of urine NOS", "Urinary retention NOS"),
    ("Other peritonitis", "Peritonitis (other)"),
    ("Anxiety disorder NOS", "Anxiety NOS"),
    ("Chest pain, unsp.", "Chest pain NOS"),
    ("Malignant pleural effusion", "Malig. pleural effusion"),
    ("E. coli (causative)", "E. coli"),
    ("Secondary malignant neoplasm (intra-abdominal LN)",
     "2° malig. neoplasm (LN)"),
]


def abbreviate_label(label: str) -> str:
    """Apply LABEL_ABBREV substitutions; conservative (most abbreviations
    live in short_names.csv via PatternLabelResolver). Also strips a
    trailing parenthetical detail (e.g. "...(ovary)") that would
    otherwise be cut off by the visit-box width, EXCEPT when the
    parenthetical is a deliberate anatomy-disambiguation tag from
    TOKEN_LABEL_OVERRIDES (those carry signal we must preserve)."""
    out = label
    for src, dst in LABEL_ABBREV:
        if src in out:
            out = out.replace(src, dst)
    # Skip the parenthetical strip if the label ends with a known
    # disambiguation tag (a paren-wrapped suffix from any of our
    # registered TOKEN_LABEL_OVERRIDES). This keeps "Sec. malig.
    # (intrathoracic LN)" from getting truncated to "Sec. malig.".
    _PROTECTED_SUFFIXES = tuple(
        v[v.rfind("("):] for v in TOKEN_LABEL_OVERRIDES.values()
        if "(" in v and v.rstrip().endswith(")")
    )
    if any(out.rstrip().endswith(suf) for suf in _PROTECTED_SUFFIXES):
        return out.rstrip()
    # Strip a trailing parenthetical specifier that adds detail past
    # the main clinical concept. We strip ANY trailing parenthetical
    # (terminated OR unterminated, with OR without ellipsis content)
    # so the visit-box display never shows artefacts like:
    #   "Fam Hx malig. neoplasm (ovary)"     → strip "(ovary)"
    #   "Fam Hx malig. neoplasm (ova…)"      → strip "(ova…)"
    #   "Peritoneal drainage (Re…"           → strip "(Re…"
    #   "Foo (a, b, c) (extra detail)"       → strip "(extra detail)"
    # The aggressive form is safe for visit-box display because the
    # parenthetical is always decorative specifier text.
    import re
    out = re.sub(r"\s*\([^()]*\)?\s*$", "", out).rstrip()
    return out


# ──────────────────────────────────────────────────────────────────────────
# Pattern matching with positions
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """One matched pattern instance for this patient."""
    pattern_str: str
    pattern_readable: str
    pattern_type: str           # cross_visit | single_visit_cooccurrence | single_token
    matched_blocks: List[Tuple[int, int]]   # [(itemset_idx, visit_idx), ...]
    claimed_positions: Set[Tuple[int, str]] # set of (pos, token_str) in IG
    sigma_ig: float                         # raw sum of IG over claimed_positions
    net_sigma_ig: float = 0.0               # set during greedy walk
    net_positions: Set[Tuple[int, str]] = field(default_factory=set)


def _classify_pattern(pat_str: str) -> str:
    blocks = pat_str.split(" -> ")
    if len(blocks) > 1:
        return "cross_visit"
    if blocks[0].strip().startswith("{"):
        return "single_visit_cooccurrence"
    return "single_token"


def _claim_positions(pattern_blocks, matched_alignment, df_ig_indexed):
    """Given pattern's [frozenset, frozenset, ...] decomposition, the matched
    alignment [(pi, vi), ...], and an IG dataframe indexed by (visit_idx,
    token_str) → list of positions, return the set of (pos, token) pairs the
    pattern claims and the sum of IG over them."""
    claimed: Set[Tuple[int, str]] = set()
    sigma = 0.0
    for pi, vi in matched_alignment:
        for tok in pattern_blocks[pi]:
            for pos, ig in df_ig_indexed.get((vi, tok), []):
                claimed.add((pos, tok))
                sigma += ig
    return claimed, sigma


def match_patterns_with_positions(df_ig: pd.DataFrame,
                                    salient_visit_blocks,
                                    df_patterns: pd.DataFrame,
                                    fmt_pat) -> List[PatternMatch]:
    """For every pattern in the mining dictionary that matches the patient's
    salient blocks, build a PatternMatch with claimed positions + sigma_IG."""
    # Build lookup: (visit_idx, token_str) -> list of (pos, ig_signed)
    sal_only = df_ig[df_ig["is_salient"] == True]  # noqa: E712
    by_vt = defaultdict(list)
    for _, row in sal_only.iterrows():
        by_vt[(int(row["visit_idx"]), row["token_str"])].append(
            (int(row["pos"]), float(row["ig_signed"])))

    matched_basic = _match_all_patterns(df_patterns, salient_visit_blocks,
                                          is_episode=True)
    # _match_all_patterns gives (pat_str, odds, support, tokens), but no
    # alignment to specific visit blocks. Recover alignment via _match_subpattern.
    out: List[PatternMatch] = []
    for pat_str, _, _, _ in matched_basic:
        ep = _parse_episode_pattern(pat_str)
        ep_fsets = [frozenset(s) for s in ep]
        align = _match_subpattern(ep_fsets, salient_visit_blocks)
        if align is None:
            continue
        claimed, sigma = _claim_positions(ep_fsets, align, by_vt)
        if not claimed:
            continue  # pattern matched at block level but no salient positions
        out.append(PatternMatch(
            pattern_str=pat_str,
            pattern_readable=fmt_pat(pat_str),
            pattern_type=_classify_pattern(pat_str),
            matched_blocks=align,
            claimed_positions=claimed,
            sigma_ig=sigma,
        ))
    return out


def greedy_assignment(matches: List[PatternMatch]) -> List[PatternMatch]:
    """Chronological greedy cover that produces a strict POSITION-LEVEL
    PARTITION of the patient's salient tokens.

    Sort priority (earliest first):
      1. earliest visit_idx in matched_blocks (chronological priority)
      2. number of visit blocks DESC (longer cross-visit patterns first)
      3. |sigma_ig| DESC (within ties, larger attribution wins)

    Each pattern claims only positions not already claimed by a
    higher-priority predecessor. Patterns fully covered by predecessors
    are dropped. Bars in the resulting waterfall sum to the salient
    Σ IG (and combined with the non-salient bucket, to the total Σ IG
    by Captum's completeness axiom on the logit output).
    """
    def _sort_key(m: PatternMatch):
        if not m.matched_blocks:
            return (10**9, 0, 0.0)  # sentinel: lowest priority
        earliest = min(vi for (_, vi) in m.matched_blocks)
        n_blocks = len(set(vi for (_, vi) in m.matched_blocks))
        return (earliest, -n_blocks, -abs(m.sigma_ig))
    matches_sorted = sorted(matches, key=_sort_key)
    claimed: Set[Tuple[int, str]] = set()
    keep: List[PatternMatch] = []
    for m in matches_sorted:
        new_pos = m.claimed_positions - claimed
        if not new_pos:
            continue
        m.net_positions = new_pos
        keep.append(m)
        claimed |= new_pos
    return keep


def backfill_net_sigma(matches: List[PatternMatch],
                        ig_signed_by_pos: dict) -> None:
    """Once net_positions is set, compute net_sigma_ig = sum of IG over those
    positions. Updates in place."""
    for m in matches:
        m.net_sigma_ig = sum(ig_signed_by_pos.get(pos, 0.0)
                              for (pos, _) in m.net_positions)


# ──────────────────────────────────────────────────────────────────────────
# Visit time fetcher (existing)
# ──────────────────────────────────────────────────────────────────────────

def fetch_visit_times(pid: str, hadm: str, mimic_dir: str):
    pkls = sorted(glob.glob(os.path.join(mimic_dir, ".atop_cache", "samples_*.pkl")))
    visit_hadm_ids = None
    for p in pkls:
        with open(p, "rb") as f:
            data = pickle.load(f)
        match = [s for s in data if str(s["patient_id"]) == pid
                                 and str(s["index_hadm_id"]) == hadm]
        if match and "visit_hadm_ids" in match[0]:
            visit_hadm_ids = [str(h) for h in match[0]["visit_hadm_ids"]]
            break
    if not visit_hadm_ids:
        return None, None
    adm = pd.read_csv(
        os.path.join(mimic_dir, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        dtype={"subject_id": str, "hadm_id": str},
        parse_dates=["admittime", "dischtime"],
    )
    # Day offsets for the visits in the input sequence
    sub = adm[adm["hadm_id"].isin(visit_hadm_ids)].sort_values("admittime").reset_index(drop=True)
    if sub.empty:
        return None, None
    t0 = sub["admittime"].iloc[0]
    sub["visit_idx"] = range(len(sub))
    sub["days_since_v1"] = (sub["admittime"] - t0).dt.total_seconds() / 86400.0

    # Find the next admission for this patient AFTER the index hadm
    pat_full = adm[adm["subject_id"] == str(pid)].sort_values("admittime").reset_index(drop=True)
    next_adm = None
    if hadm in pat_full["hadm_id"].values:
        idx_disch = pat_full[pat_full["hadm_id"] == hadm]["dischtime"].iloc[0]
        idx_pos = pat_full.index[pat_full["hadm_id"] == hadm].tolist()[0]
        if idx_pos + 1 < len(pat_full):
            n = pat_full.iloc[idx_pos + 1]
            days_to_next = (n["admittime"] - idx_disch).days
            next_adm = {"hadm_id": n["hadm_id"], "days_post": int(days_to_next)}
    return sub, next_adm


# ──────────────────────────────────────────────────────────────────────────
# Rendering, TOP PANEL: timeline with arcs
# ──────────────────────────────────────────────────────────────────────────

def _draw_arc(ax, x0, x1, y_anchor, y_apex, color, lw, alpha):
    """Draw a quadratic Bezier arc from (x0, y_anchor) → (x1, y_anchor) with
    apex at y_apex. Returns the apex point for label placement."""
    verts = [(x0, y_anchor), ((x0 + x1) / 2, y_apex), (x1, y_anchor)]
    codes = [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color,
                                  lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    qx = 0.25 * x0 + 0.75 * x1
    qy = 0.25 * y_anchor + 0.5 * y_apex + 0.25 * y_anchor
    ax.annotate("", xy=(x1, y_anchor), xytext=(qx, qy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, alpha=alpha, mutation_scale=12),
                zorder=2)
    return ((x0 + x1) / 2, y_apex)


def aggregate_arcs_per_source(cross_visit_kept, idx_x):
    """Sankey-lite: per source visit, aggregate cross-visit patterns to one
    ribbon. Returns dict {src_visit: {n_patterns, sigma_signed, sigma_abs,
    top_pattern_label}}."""
    by_src = defaultdict(lambda: {"n": 0, "signed": 0.0, "abs": 0.0,
                                     "top_label": None, "top_abs": 0.0})
    for m in cross_visit_kept:
        src_visits = [vi for (_, vi) in m.matched_blocks if vi != idx_x]
        if not src_visits:
            continue
        src = min(src_visits)
        d = by_src[src]
        d["n"] += 1
        d["signed"] += m.net_sigma_ig
        d["abs"]    += abs(m.net_sigma_ig)
        if abs(m.net_sigma_ig) > d["top_abs"]:
            d["top_abs"]   = abs(m.net_sigma_ig)
            d["top_label"] = m.pattern_readable
    return dict(by_src)


def render_timeline(ax, n_visits, visit_times, next_adm,
                      visit_top_tokens, y_true, y_prob, fmt_short):
    """Render the visit timeline panel.

    All visit boxes are uniform width (legibility wins over strict
    proportionality). Initial x-positions are proportional to days, then
    a left-to-right sweep enforces a minimum inter-box gap by pushing
    boxes apart as needed. The real-days timeline below sits directly
    under the boxes with tick marks aligned to box centers; the day
    labels at each tick show the ACTUAL day values (so the reader sees
    true gaps in the numbers, even though visual spacing is adjusted).
    A short vertical connector ties each tick to its box.

    Args:
      visit_top_tokens: dict {vi: [(token_readable, ig_signed), ...]} top-3
    """
    # ── Geometry: clinical-timeline style ─────────────────────────
    # No boxes. Each visit is a left-justified text block (V_n header
    # in bold + 3 token rows) anchored at its day-proportional x.
    # Text blocks alternate ABOVE / BELOW the horizontal timeline.
    # A short vertical connector links each text block to the dot on
    # the timeline. Day numbers sit on the OPPOSITE side of the
    # timeline from the text block (so each event's title + body + day
    # form a tight visual cluster).
    TEXT_BLOCK_W = 2.2          # estimated text-block width (data units);
                                #   tighter than before so V1/V2/V3 cluster
                                #   visits stay near their day-proportional
                                #   positions and only need minimal push.
    TEXT_BLOCK_H = 1.55         # estimated text-block height
                                #   (V_n header + 3 token rows)
    GAP_FROM_TIMELINE = 1.60    # gap between timeline and FIRST token,
                                # sized so the "Vx · Day X" label (which
                                # sits within DAY_MARGIN+DAY_LABEL_H of
                                # the line) has clear space above/below it
                                # before the token block begins.
    Y_TIMELINE = 0.0
    Y_ABOVE = Y_TIMELINE + GAP_FROM_TIMELINE   # text-block BOTTOM if above
    Y_BELOW = Y_TIMELINE - GAP_FROM_TIMELINE   # text-block TOP if below
    PROP_SCALE = 13.0           # total horizontal span (data units)

    idx_x = n_visits - 1

    # ── Day data ──────────────────────────────────────────────────────
    if visit_times is not None and len(visit_times) == n_visits:
        days = visit_times["days_since_v1"].values.astype(float)
    else:
        days = np.arange(n_visits, dtype=float) * 100.0
    day_max = float(max(days[-1], 1.0))

    # ── Compute x positions PROPORTIONAL to days + above/below side ──
    # Algorithm: for each visit, prefer ABOVE; fall back to BELOW; if
    # both sides have a recent same-side text block within TEXT_BLOCK_W,
    # push the visit's x rightward just enough to clear the
    # least-recent same-side block.
    if day_max > 0:
        x_positions = (days / day_max * PROP_SCALE).astype(float).tolist()
    else:
        x_positions = np.linspace(0, PROP_SCALE, n_visits).tolist()

    # Compress ONLY the single largest inter-visit gap, by half. All
    # other gaps stay strictly proportional to real days. Rationale: a
    # patient with a multi-year skip between two visits otherwise wastes
    # half the figure on whitespace, but we don't want to lie about the
    # relative magnitudes of the OTHER (within-cluster) gaps.
    if n_visits >= 2:
        gaps = [x_positions[i] - x_positions[i-1] for i in range(1, n_visits)]
        max_gap_idx = 1 + int(np.argmax(gaps))
        max_gap = x_positions[max_gap_idx] - x_positions[max_gap_idx - 1]
        new_gap = max_gap * 0.35
        delta = max_gap - new_gap
        for j in range(max_gap_idx, n_visits):
            x_positions[j] -= delta

    sides = ["above"] * n_visits
    last_x_on_side = {"above": -1e9, "below": -1e9}

    # Side-preference order: start with BELOW for V1 so the alternation
    # walks V1↓ V2↑ V3↓ V4↑ V5↓ … and the INDEX visit (V_last, where
    # the prediction is made + readmission marker sits) lands on the
    # BOTTOM side. The reader's eye naturally ends on the index visit's
    # tokens + the ŷ badge below them, with the readmission triangle
    # marker on the timeline above the cluster. Keeps the figure's
    # "story end" visually anchored.
    # We HARD-FORCE the index visit to "below" (regardless of collision
    # checks) so the readmission text, which always lives above the
    # line at the right edge, never competes with V_last's tokens for
    # the same real estate.
    if n_visits % 2 == 1:
        preferred = ("below", "above")     # odd n → V1 below, V_last below
    else:
        preferred = ("above", "below")     # even n → V1 above, V_last below

    for i in range(n_visits):
        xp = x_positions[i]
        if i == n_visits - 1:                  # index visit: force below
            chosen = "below"
        elif i == n_visits - 2:                # visit just before index:
            # force ABOVE so the alternation reads naturally ↑↓ at the
            # right end (V_{last-1} above, V_last below) without the
            # compressed-rightmost-gap pushing two visits to the same
            # side.
            chosen = "above"
        else:
            chosen = None
            for side in preferred:
                if (xp - last_x_on_side[side]) >= TEXT_BLOCK_W:
                    chosen = side
                    break
        if chosen is None:
            chosen = min(("above", "below"),
                         key=lambda s: last_x_on_side[s])
            xp = last_x_on_side[chosen] + TEXT_BLOCK_W
            x_positions[i] = xp
        sides[i] = chosen
        last_x_on_side[chosen] = xp

    # Readmission marker, placed on the timeline VERY CLOSE to V5,
    # since post-discharge gap is days (not months). Just a small
    # visible offset so the dot doesn't overlap V5's dot.
    # Push the readmission marker further right of V_idx so its label
    # has its own column and doesn't crowd the V_idx day-label cluster.
    r30_x = x_positions[idx_x] + 0.95

    # ── Draw the horizontal timeline ─────────────────────────────────
    # Thicker, grayer line so it reads as an axis rather than a content
    # element, keeps the eye on the token text, not the rule.
    # The xlim extends LEFT of the first visit by TOKEN_BLOCK_OFFSET +
    # padding so V1's left-justified tokens have room without being
    # clipped at the canvas edge.
    # Keep in sync with TOKEN_BLOCK_OFFSET set in the per-visit loop
    # below, duplicated here because the timeline must be drawn before
    # the loop begins.
    _LEFT_PAD_FOR_TOKENS = 0.40 + 0.30
    x_left  = x_positions[0] - _LEFT_PAD_FOR_TOKENS
    # Room on the right for the stacked two-line readmission label
    # ("Patient readmitted / N days post-discharge").
    x_right = r30_x + 3.5
    ax.plot([x_positions[0], r30_x - 0.05],
            [Y_TIMELINE, Y_TIMELINE],
            color="#999999", lw=2.6, zorder=2, solid_capstyle="round")

    # ── Per-visit dot + connector + day label + text block ──────────
    line_dy = 0.52      # vertical spacing between token rows, kept
                        # tight so each visit's 3-token block reads as
                        # a compact group (was 0.72; user wants less
                        # whitespace inside each block).
    MAX_LEN = 46
    BODY_FS   = 13.0    # token row font size, larger so tokens are
                        # readable at any reasonable zoom.
    DAY_FS    = 12.0    # day label font size, bold + larger so it
                        # reads as an axis tick, not as inline text.
    DAY_COLOR = "#555555"   # medium gray + bold, clearly readable as
                            # axis labels but still secondary to the
                            # token text which uses RISK / PROT colors.
    DAY_MARGIN = 0.50       # vertical margin between timeline and day
                            # label, kept tight so "Vx · Day X" sits
                            # close to the axis, reading as an axis
                            # tick label rather than free-floating text.
    DAY_LABEL_H = 0.35      # estimated height of the day label band;
                            # used to start connector stems past the
                            # label so the stem doesn't bisect the text.
    TOKEN_BLOCK_OFFSET = 0.40  # how far LEFT of the visit's dot to
                            # left-justify the token text. Sized so the
                            # ▲/▼ glyph + first ~3 chars of the label
                            # sit directly under the visit circle, and
                            # the rest of the label extends to the
                            # right. Visually, the connector stem now
                            # lines up with the GLYPH column, not the
                            # text midpoint.
    CONNECTOR_PAD = 0.10    # how far past the outermost token the
                            # connector stem extends (so the line visibly
                            # spans the full token block height)

    for i in range(n_visits):
        is_idx = (i == idx_x)
        x = x_positions[i]
        side = sides[i]
        toks = visit_top_tokens.get(i, [])

        # Visit marker, large hollow "station" circle with "Vx"
        # centered inside. Two-ring style: an outer subtle halo + an
        # inner bold edge, so the circle reads as a UI badge against
        # the timeline rule. Same size + color for all visits; the
        # index visit's emphasis comes from the ŷ badge under its
        # tokens.
        # Outer halo (slightly larger, very light gray fill, masks
        # the timeline rule through the circle)
        ax.scatter([x], [Y_TIMELINE], s=1500, facecolors="white",
                    edgecolors="#dddddd", linewidths=1.0,
                    marker="o", zorder=5)
        # Inner badge (the visual circle itself). The index admission
        # gets a slightly warmer gray fill (with a subtle warm-tone
        # edge) so the reader can instantly see WHICH admission is
        # the prediction point, without needing the caption alone to
        # do that work.
        if is_idx:
            _fill, _edge = "#fbe7c8", "#b87333"
        else:
            _fill, _edge = "#f0f0f0", "#333333"
        ax.scatter([x], [Y_TIMELINE], s=1200, facecolors=_fill,
                    edgecolors=_edge, linewidths=2.4,
                    marker="o", zorder=6)
        # Vx label, kept at the same size as before; only the
        # surrounding circle grows.
        ax.text(x, Y_TIMELINE, f"V{i+1}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=("#7a4a13" if is_idx else "#333333"),
                zorder=7)

        # Day label, ALWAYS just below the dot, in teal bold (matches
        # the reference clinical-timeline style). Short label so close
        # clusters (e.g., Day 0 / Day 35) don't visually collide. V_n
        # identifier is implicit by left-to-right order; Panel B
        # references like "V1→V2:..." remain unambiguous.
        # Day label includes the visit identifier (V1, V2, …) so the
        # waterfall's "V1→V2:" prefixes have an unambiguous anchor in
        # Panel A. Place the day label on the OPPOSITE side from the
        # tokens, so tokens above → "Vx · Day X" below the line,
        # tokens below → "Vx · Day X" above the line. This keeps the
        # day label off any connector stem and gives clear visual
        # separation between the timeline anchor and the token rows.
        # EXCEPTION: at the index visit, the above-right region is
        # reserved for the "Readmission" outcome marker, so a tokens-
        # below index visit must put its day label below too (a small
        # asymmetry that prevents the index-visit day label from
        # colliding with the readmission text).
        # Day label, visit identifier now lives INSIDE the circle, so
        # this label only shows the elapsed-day anchor.
        day_text = f"Day {int(days[i])}"
        if is_idx:
            # Index visit: day label ALWAYS on top (above the line),
            # regardless of which side the tokens went. The "Readmission"
            # marker text also sits above and slightly to the right, so
            # the two coexist as a small cluster at the figure's terminus.
            day_y, day_va = Y_TIMELINE + DAY_MARGIN, "bottom"
        elif side == "above":
            day_y, day_va = Y_TIMELINE - DAY_MARGIN, "top"
        else:
            day_y, day_va = Y_TIMELINE + DAY_MARGIN, "bottom"
        ax.text(x, day_y, day_text,
                ha="center", va=day_va, fontsize=DAY_FS,
                color=DAY_COLOR, fontweight="bold", zorder=5)
        # Index-admission tag, small italic line just past the day
        # label, in the warm tone that matches the index-visit circle
        # fill. Identifies which admission is the prediction point.
        if is_idx:
            # Place below the day label (which itself sits above the
            # timeline for the index visit).
            _idx_y = day_y + (0.30 if day_va == "bottom" else -0.30)
            _idx_va = day_va
            ax.text(x, _idx_y, "Index admission",
                    ha="center", va=_idx_va, fontsize=DAY_FS - 1.5,
                    color="#7a4a13", fontstyle="italic",
                    fontweight="bold", zorder=5)

        # Token list, left-justified at dot's x. Above OR below the
        # timeline based on alternation. Each line:
        #   "+0.10  Ascites"
        # No bold V_n header; the day label IS the section identifier.
        # Token rows per visit are now variable (surfaced + up to TAIL_K
        # tail tokens, capped at PER_VISIT_CAP=5). body_n drives the
        # geometry of the block + connector stem.
        body_n   = len(toks)
        # Connector stem starts just past the dot, the day label sits
        # on the OPPOSITE side from tokens, so the stem has a clear
        # path from the timeline up/down to the token block.
        stem_inner = 0.10
        # Connector stem STOPS WELL BEFORE the nearest token's text body
        # so the line never visually overlaps the bottommost above-token
        # (or topmost below-token). Sized to clear the ~0.45-unit text
        # height of a 13pt body row + a small visual breathing gap.
        STEM_TO_TOKEN_GAP = 0.55
        if side == "above":
            # Tokens stack DOWNWARD from the top of the block toward
            # the timeline; reading top→bottom keeps the most
            # impactful (largest |IG|) token at the top.
            top_y = Y_ABOVE + body_n * line_dy
            # Connector, from just above the timeline UP to just
            # BELOW the first token row. The stem stops short of the
            # token text so the line never visually crosses the
            # token block.
            stem_top = top_y - (body_n - 1) * line_dy - STEM_TO_TOKEN_GAP
            ax.plot([x, x],
                    [Y_TIMELINE + stem_inner, stem_top],
                    color="#888", lw=0.7, zorder=2)
            for j, (lab, ig) in enumerate(toks):
                color = RISK_COLOR if ig > 0 else PROT_COLOR
                glyph = "▲" if ig > 0 else "▼"
                short_lab = abbreviate_label(lab)
                if len(short_lab) > MAX_LEN:
                    short_lab = short_lab[:MAX_LEN - 1] + "…"
                ax.text(x - TOKEN_BLOCK_OFFSET,
                        top_y - j * line_dy,
                        f"{glyph}  {short_lab}",
                        ha="left", va="top", fontsize=BODY_FS,
                        color=color, zorder=5)
            # (ŷ badge removed from Panel A, the prediction value
            # lives in Panel B's right anchor "Σ IG / predicted ŷ".)
        else:
            # Below block, tokens stack DOWNWARD past Y_BELOW (which
            # is itself GAP_FROM_TIMELINE units past the timeline, so
            # there's room for the opposite-side day label without
            # collision). The index visit's day label now sits ABOVE
            # the line with the readmission marker, so the below-block
            # geometry is uniform across all below-side visits.
            below_top = Y_BELOW - 0.10
            # Connector, from just below the timeline DOWN to just
            # ABOVE the first (topmost) token row. The stem stops short
            # of the token text so the line never visually crosses the
            # token block.
            ax.plot([x, x],
                    [Y_TIMELINE - stem_inner,
                     below_top + STEM_TO_TOKEN_GAP],
                    color="#888", lw=0.7, zorder=2)
            for j, (lab, ig) in enumerate(toks):
                color = RISK_COLOR if ig > 0 else PROT_COLOR
                glyph = "▲" if ig > 0 else "▼"
                short_lab = abbreviate_label(lab)
                if len(short_lab) > MAX_LEN:
                    short_lab = short_lab[:MAX_LEN - 1] + "…"
                ax.text(x - TOKEN_BLOCK_OFFSET,
                        below_top - j * line_dy,
                        f"{glyph}  {short_lab}",
                        ha="left", va="top", fontsize=BODY_FS,
                        color=color, zorder=5)
            # (ŷ badge removed from Panel A, the prediction value
            # lives in Panel B's right anchor "Σ IG / predicted ŷ".)

    # ── Readmission marker (same style as a day tick) ─────────────
    # Just another dot on the timeline, very close to V5. Label below
    # in the same gray-day style for the day number ("Day 913"), with
    # "Readmission" in red bold on the next line so reviewers see the
    # outcome at a glance.
    r30_color = R30_RISK_COLOR if y_true == 1 else R30_PROT_COLOR
    # Two-color label: the outcome itself in the outcome color
    # (dark crimson / dark navy), the temporal qualifier in a neutral
    # muted gray so it reads as secondary anchoring info, not as part
    # of the outcome itself.
    if y_true == 1 and next_adm is not None:
        r30_top = "Patient readmitted"
        r30_bot = f"{next_adm['days_post']} days post-discharge"
    elif y_true == 1:
        r30_top = "Patient readmitted"
        r30_bot = "within 30 days"
    else:
        r30_top = "No readmission"
        r30_bot = "within 30 days"

    # Common x anchor for the icon + the two outcome label lines,
    # centering everything on the same x guarantees the icon sits
    # directly above the text without needing glyph-width estimation.
    _R30_LABEL_X = r30_x + 1.50
    # Optional icon ABOVE the readmission text: a hospital ↔ home pin
    # graphic that gives the readmission outcome instant visual
    # meaning. Only shown when the patient WAS readmitted (the
    # "discharge home → return to hospital" semantics fit only the
    # positive case). Missing icon file degrades gracefully to no
    # icon.
    if y_true == 1:
        icon_path = os.path.join(os.path.dirname(__file__),
                                   "assets", "readmission_icon.png")
        if os.path.exists(icon_path):
            try:
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                import matplotlib.image as mpimg
                _icon = mpimg.imread(icon_path)
                # Center the icon horizontally on the SAME x-anchor
                # used by the "Patient readmitted" text (rendered
                # immediately below with ha="center"), so the icon
                # always sits exactly above the text midpoint
                # regardless of glyph-width estimation.
                _imbox = OffsetImage(_icon, zoom=0.62)
                _ab = AnnotationBbox(
                    _imbox,
                    (_R30_LABEL_X, Y_TIMELINE + 1.15),
                    frameon=False, box_alignment=(0.5, 0.5),
                    zorder=5)
                ax.add_artist(_ab)
            except Exception:
                pass   # any failure: skip the icon, render text only
    # Readmission marker, distinct SHAPE and color from both the
    # visit circles AND the per-token ▲/▼ glyphs. We use a filled
    # diamond ("D" marker) in dark crimson (readmission) or navy
    # (no readmission). The diamond is geometric and clearly distinct
    # from triangles, so the reader doesn't misread it as a token.
    ax.scatter([r30_x], [Y_TIMELINE], s=300, color=r30_color,
                marker="D", edgecolors="white", linewidths=1.4,
                zorder=6)
    # Outcome label, TWO LINES, stacked, vertically straddling the
    # timeline axis and placed to the RIGHT of the diamond marker.
    # Top line: outcome ("Patient readmitted") in the outcome color.
    # Bottom line: temporal qualifier ("N days post-discharge") in
    # neutral gray so it reads as secondary anchoring info rather
    # than competing with the outcome.
    _stack_dy = 0.22
    ax.text(_R30_LABEL_X, Y_TIMELINE + _stack_dy, r30_top,
            ha="center", va="center", fontsize=DAY_FS + 0.5,
            fontweight="bold", color=r30_color, zorder=5)
    ax.text(_R30_LABEL_X, Y_TIMELINE - _stack_dy, r30_bot,
            ha="center", va="center", fontsize=DAY_FS,
            color="#666666", zorder=5, style="italic")

    # (No "days from V1" caption, day labels are self-evident.)

    # ── Axis cosmetics ────────────────────────────────────────────────
    # Compute y bounds from the highest above-text and lowest below-text.
    # Above-side max y ≈ Y_ABOVE + 3*line_dy + CONNECTOR_PAD + ŷ-badge room
    # Below-side min y ≈ Y_BELOW - 0.10 - 2*line_dy - CONNECTOR_PAD - ŷ-badge room
    # y bounds must accommodate the TALLEST visit's token block on
    # each side. Token count per visit is now variable (up to
    # PER_VISIT_CAP=5 rows), so derive max-rows from the actual
    # visit_top_tokens data on each side.
    above_max = max(
        (len(visit_top_tokens.get(i, [])) for i in range(n_visits)
         if sides[i] == "above"),
        default=3)
    below_max = max(
        (len(visit_top_tokens.get(i, [])) for i in range(n_visits)
         if sides[i] == "below"),
        default=3)
    y_top = Y_ABOVE + above_max * line_dy + CONNECTOR_PAD + 0.55
    y_bot = (Y_BELOW - 0.10) - (below_max - 1) * line_dy - CONNECTOR_PAD - 0.55
    ax.set_xlim(x_left - 0.15, x_right + 0.15)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────
# Rendering, BOTTOM PANEL: greedy waterfall
# ──────────────────────────────────────────────────────────────────────────

def _draw_shap_chevron(ax, x_start, x_end, y, height, color, tip_abs,
                          x_axis_to_data_units: float = 1.0):
    """Draw a SHAP-style chevron arrow bar.

    Starting side (x_start): flat vertical line.
    Outgoing side (x_end):   subtle convex chevron tip pointing in the
                             direction of contribution. Tip size is
                             height-proportional (constant visual angle
                             per bar), matches SHAP's force-plot look.
    """
    h = height
    half_h = h / 2
    # Height-based tip: matplotlib y-units are bar height units; convert
    # to data-x-units using the supplied aspect-correction factor.
    # Default: tip = 0.30 * h in y-units → translates to data-x via the
    # caller's transform. We use a fixed fraction of total span as a
    # safe upper bound so very narrow bars don't end up all tip.
    tip = min(tip_abs, abs(x_end - x_start) * 0.20)
    if x_end >= x_start:
        # Positive bar, subtle right-pointing tip
        verts = [
            (x_start,            y - half_h),
            (x_start,            y + half_h),
            (x_end - tip,        y + half_h),
            (x_end,              y),
            (x_end - tip,        y - half_h),
        ]
    else:
        # Negative bar, subtle left-pointing tip
        verts = [
            (x_start,            y - half_h),
            (x_start,            y + half_h),
            (x_end + tip,        y + half_h),
            (x_end,              y),
            (x_end + tip,        y - half_h),
        ]
    poly = mpatches.Polygon(verts, closed=True, facecolor=color,
                              edgecolor=color, lw=0.6, zorder=3)
    ax.add_patch(poly)


def render_greedy_waterfall(ax_bars, ax_labels, kept_matches, residual_sigma,
                              residual_n_tokens, fmt_pat,
                              preserve_order: bool = False,
                              nonsalient_sigma: float = 0.0,
                              nonsalient_count: int = 0,
                              total_signed_ig: float = 0.0,
                              y_prob: float = 0.0):
    """Horizontal greedy waterfall, split across two axes.

    Bars (top-to-bottom):
      - Each kept pattern (greedy marginal IG over newly-claimed positions)
      - Residual unmatched salient tokens
      - Non-salient tokens (collapsed), closes the gap to the prediction
      Sum of bar lengths = total Σ IG = ŷ(x) − ŷ(baseline) (Captum
      completeness axiom). The cumulative marker shows this total.

    ax_bars: axes that hosts the SHAP-style chevron bars
    ax_labels: axes (left of ax_bars, axis hidden) for left-justified labels
    """
    rows = []  # (label, value, color)
    if preserve_order:
        sorted_kept = list(kept_matches)
    else:
        sorted_kept = sorted(kept_matches, key=lambda m: -abs(m.net_sigma_ig))
    for m in sorted_kept:
        # Visit-reach prefix: "V3→V5: ..." for cross-visit, "V5: ..." for
        # single-visit (covers both single-token and co-occurrence).
        visits_used = sorted(set(vi for (_, vi) in m.matched_blocks)) if m.matched_blocks else []
        if not visits_used:
            prefix = ""
        elif len(visits_used) == 1:
            prefix = f"V{visits_used[0]+1}: "
        else:
            prefix = f"V{visits_used[0]+1}→V{visits_used[-1]+1}: "
        label = abbreviate_label(prefix + m.pattern_readable)
        color = RISK_COLOR if m.net_sigma_ig > 0 else PROT_COLOR
        rows.append((label, m.net_sigma_ig, color))
    # Residual, salient positions not claimed by any top-N named pattern.
    # NB: this bucket combines two distinct categories: (a) salient tokens
    # with no eligible mined pattern (e.g. rare codes below min_support),
    # and (b) salient tokens claimed by kept patterns ranked below the
    # waterfall display cap. Labeled "Other salient" rather than
    # "Residual ... unmatched" to avoid implying category (b) was unmatched.
    if abs(residual_sigma) > 1e-9:
        color = RISK_COLOR if residual_sigma > 0 else PROT_COLOR
        rows.append((f"Other salient tokens ({residual_n_tokens})",
                     residual_sigma, color))
    # Non-salient bucket: closes the gap so bars sum to total Σ IG ≈
    # ŷ − baseline. Without it the waterfall only shows ~80% of |IG|.
    if abs(nonsalient_sigma) > 1e-9:
        color = RISK_COLOR if nonsalient_sigma > 0 else PROT_COLOR
        rows.append((f"Non-salient tokens ({nonsalient_count})",
                     nonsalient_sigma, color))

    # Plot from top to bottom (caller controls order via preserve_order)
    n = len(rows)
    y_positions = list(range(n - 1, -1, -1))

    # Compute total span for chevron-tip sizing (proportional to x range)
    total_signed = sum(v for (_, v, _) in rows)
    span = max(abs(sum(min(0, v) for (_, v, _) in rows)),
                 abs(sum(max(0, v) for (_, v, _) in rows)),
                 abs(total_signed),
                 1e-6)
    # Chevron tip absolute size: small, subtle (SHAP-style). Capped at
    # 20% of the bar's own length in _draw_shap_chevron.
    tip_abs = max(span * 0.005, 0.002)
    # Bars take up MORE of each row-unit so the inter-row gap shrinks
    # visually (rows feel tightly packed instead of widely spaced).
    bar_h = 0.62

    # Cumulative chained SHAP-style chevron bars (on ax_bars)
    cum = 0.0
    for y, (label, val, color) in zip(y_positions, rows):
        x_start = cum
        x_end = cum + val
        _draw_shap_chevron(ax_bars, x_start, x_end, y, bar_h, color, tip_abs)
        # Annotation: signed value placed beyond the tip
        text_x = x_end + (tip_abs * 1.2 if val > 0 else -tip_abs * 1.2)
        ha = "left" if val > 0 else "right"
        ax_bars.text(text_x, y, f"{val:+.3f}", ha=ha, va="center",
                fontsize=12, color="#333333", zorder=4)
        cum += val

    # IG was integrated on the logit output (atop/attribution/ig.py),
    # so by Captum's completeness:
    #   Σ IG_i  =  logit(ŷ)  −  logit(ŷ_baseline_input)
    # We back out the baseline (PAD-input) probability:
    import math
    eps = 1e-6
    p_clip = min(max(y_prob, eps), 1 - eps)
    logit_y = math.log(p_clip / (1 - p_clip))
    logit_base = logit_y - cum
    y_base = 1.0 / (1.0 + math.exp(-logit_base))

    # ── x-axis anchoring (right endpoint only) ────────────────────
    # Path A: drop the "baseline ŷ = ..." left anchor entirely. The
    # IG reference (all-PAD input) doesn't correspond to a meaningful
    # clinical state, so labeling x = 0 as "baseline" was misleading.
    # We keep the cumulative bars (additivity is mathematically real
    # via Captum's completeness axiom) but only anchor the right
    # endpoint, where the chain lands on the actual prediction.
    #
    # x = 0    : zero contribution (visual chain start; no claim about
    #            what model state it corresponds to)
    # x = Σ IG : the cumulative contribution that yields ŷ_pred
    ax_bars.axvline(0, color="black", linewidth=0.6, zorder=2)
    # Predicted-ŷ vertical: black + dotted so the cumulative anchor
    # reads clearly against the bars.
    ax_bars.axvline(cum, color="black", linestyle=":",
                     linewidth=1.0, zorder=2)
    # Right anchor, two-line label placed to the RIGHT of the dotted
    # cumulative line. Top line states the math (Σ IG total); bottom
    # line states the clinical translation (predicted ŷ). Carries both
    # the additivity claim and the prediction at the natural endpoint
    # of the chain.
    label_x_offset = max((max(0, cum)) * 0.03, tip_abs * 4)
    ax_bars.text(cum + label_x_offset, n + 0.3,
                  f"Σ IG = {cum:+.3f}\npredicted ŷ = {y_prob:.2f}",
                  ha="left", va="bottom", fontsize=12,
                  fontweight="bold", color="#333333",
                  linespacing=1.2)

    # Bars-axis cosmetics, Δlogit scale (IG was integrated on logits).
    # Axis label matches the Figure 2 caption in the manuscript exactly.
    ax_bars.set_yticks([])
    # Axis label is intentionally plain, the right-endpoint anchor
    # (predicted ŷ) tells the reader where the chain lands. The caption
    # explains the logit-vs-probability distinction in full; we don't
    # repeat it in the axis label to avoid jargon collision.
    ax_bars.set_xlabel(
        "Cumulative Σ IG (logit-space contribution to ŷ)",
        fontsize=12.5)
    ax_bars.tick_params(axis="x", labelsize=11)
    ax_bars.grid(True, axis="x", color="#eeeeee", linewidth=0.5, zorder=1)
    for s in ("top", "right", "left"):
        ax_bars.spines[s].set_visible(False)
    ax_bars.set_ylim(-0.5, n + 1.0)

    # Tighten x-axis to data range with light padding (no random stretch)
    cum_track = 0.0
    cum_min = 0.0
    cum_max = 0.0
    for (_, val, _) in rows:
        cum_track += val
        cum_min = min(cum_min, cum_track)
        cum_max = max(cum_max, cum_track)
    # Extra right padding so the chained-cumulative Δlogit dotted line
    # AND its value labels (small bars right next to it) don't clash
    # against the right edge.
    pad = max((cum_max - cum_min) * 0.06, span * 0.02)
    ax_bars.set_xlim(cum_min - pad, cum_max + pad * 5.5)

    # Labels axis: hide spines + ticks, just left-justified text
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(ax_bars.get_ylim())
    for s in ("top", "right", "left", "bottom"):
        ax_labels.spines[s].set_visible(False)
    ax_labels.set_xticks([])
    ax_labels.set_yticks([])
    for y, (label, _, _) in zip(y_positions, rows):
        ax_labels.text(0.02, y, label, ha="left", va="center",
                       fontsize=13.5, color="#222222")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure 2 (timeline + greedy waterfall)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v2")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--exemplar-key", default="main_tp",
                   choices=["main_tp", "complex_tp", "alt_tp", "confident_tn", "disagreement"])
    p.add_argument("--pid", default=None,
                   help="Direct patient_id override (requires --hadm)")
    p.add_argument("--hadm", default=None,
                   help="Direct index_hadm_id override")
    p.add_argument("--output-path", default="../manuscript/jamia_submission/figures/Figure2.png",
                   help="Auto-suffixed with _<pid>.png unless --no-id-suffix.")
    p.add_argument("--no-id-suffix", action="store_true")
    p.add_argument("--top-arcs", type=int, default=8,
                   help="Maximum cross-visit arcs to draw on the timeline.")
    p.add_argument("--top-waterfall", type=int, default=12,
                   help="Maximum patterns shown in the greedy waterfall (excluding residual).")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)

    # ── Load exemplar / direct override ───────────────────────────────
    if args.pid and args.hadm:
        pid, hadm = str(args.pid), str(args.hadm)
        preds = pd.read_csv(
            os.path.join(args.run_dir, "2_predictions/test_predictions.csv"),
            dtype={"subject_id": str, "hadm_id": str})
        pred_row = preds[(preds["subject_id"] == pid) & (preds["hadm_id"] == hadm)]
        if len(pred_row) == 0:
            sys.exit(f"  [ERROR] no prediction for pid={pid} hadm={hadm}")
        y_prob = float(pred_row["y_prob"].iloc[0])
        y_true = int(pred_row["y_true"].iloc[0])
    else:
        with open(layout.exemplars_json) as f:
            exemplar = json.load(f)[args.exemplar_key]
        pid, hadm = exemplar["patient_id"], exemplar["index_hadm_id"]
        y_prob = exemplar["y_prob"]
        y_true = int(exemplar.get("y_true", 0))

    print(f"[1] Exemplar: pid={pid} hadm={hadm} y_prob={y_prob:.3f} y_true={y_true}")

    # ── Formatters ────────────────────────────────────────────────────
    # Use the SAME short_names.csv that Fig 3 / Fig 4 use so token
    # abbreviations are consistent across all manuscript figures.
    short_names_csv = os.path.join(os.path.dirname(__file__), "short_names.csv")
    if os.path.exists(short_names_csv):
        resolver = PatternLabelResolver(args.run_dir,
                                         short_names_csv=short_names_csv,
                                         verbose=False)
    else:
        resolver = PatternLabelResolver(args.run_dir, verbose=False)
    # Inject anatomy-disambiguation overrides into the resolver's
    # short-name table so BOTH single-token rendering (Panel A) AND
    # pattern rendering (Panel B) pick them up consistently. Without
    # this, distinct ICD-10 codes that share an English phrase (e.g.,
    # C771/C775/C786 all → "Secondary malig. neoplasm") render
    # indistinguishably in Panel B's pattern labels.
    for _tok, _label in TOKEN_LABEL_OVERRIDES.items():
        resolver._short_names[_tok] = _label
    def fmt_pat(s, max_len=95):
        lab = resolver.format_pattern(s, short_names=True)
        return lab if len(lab) <= max_len else lab[:max_len-1] + "…"
    def fmt_short(t, max_len=34):
        # First check the anatomy-disambiguation override table, this
        # keeps distinct ICD-10 codes from collapsing to the same
        # English string in Panel A. Falls back to the resolver for
        # everything else.
        if t in TOKEN_LABEL_OVERRIDES:
            return TOKEN_LABEL_OVERRIDES[t]
        # Generous truncation budget so visit boxes can fit canonical
        # short names like "Drain peritoneal cavity (percutaneous)" or
        # "Family Hx malignant neoplasm (ovary)" without ellipsis.
        lab = resolver.format_token(t)
        return lab if len(lab) <= max_len else lab[:max_len-1] + "…"

    # ── IG ────────────────────────────────────────────────────────────
    df_ig_all = pd.read_csv(layout.ig_csv("test"),
                              dtype={"patient_id": str, "index_hadm_id": str})
    df_ig = df_ig_all[(df_ig_all["patient_id"] == pid)
                      & (df_ig_all["index_hadm_id"] == hadm)].copy()
    ig_signed_by_pos = dict(zip(df_ig["pos"], df_ig["ig_signed"]))
    df_sal = df_ig[df_ig["is_salient"] == True]  # noqa: E712
    print(f"[2] IG: {len(df_ig)} rows ({len(df_sal)} salient)")

    # Per-visit top-3 salient tokens, selection deferred until after
    # the greedy waterfall is computed, so we can tie-break in favor
    # of tokens that ALSO appear in Panel B (avoids cases like V4
    # where AKI NOS is in a top Panel B pattern but is rank-4 in raw
    # |IG|, displaced by a near-tied token that's NOT in Panel B).
    # visit_top_tokens is populated further down (search for [TOP3]).

    # ── Salient visit blocks + n_visits ───────────────────────────────
    with open(layout.sequences_pkl("test"), "rb") as f:
        seq_df = pickle.load(f)
    seq_df["patient_id"] = seq_df["patient_id"].astype(str)
    seq_df["index_hadm_id"] = seq_df["index_hadm_id"].astype(str)
    seq_row = seq_df[(seq_df["patient_id"] == pid)
                       & (seq_df["index_hadm_id"] == hadm)].iloc[0]
    salient_visit_blocks = list(seq_row["salient_visit_blocks"])
    n_visits = int(seq_row["n_visits"])
    print(f"[3] Sequence: n_visits={n_visits}, "
          f"salient_blocks size = {[len(b) for b in salient_visit_blocks]}")

    # ── Mining patterns ───────────────────────────────────────────────
    pat_files = sorted(glob.glob(os.path.join(layout.mining_dir, "all_scored_*.csv")),
                        key=os.path.getmtime)
    df_patterns = pd.read_csv(pat_files[-1])
    print(f"[4] Mining: {os.path.basename(pat_files[-1])} ({len(df_patterns):,} patterns)")

    # ── Match patterns + greedy assignment ────────────────────────────
    matches = match_patterns_with_positions(df_ig, salient_visit_blocks,
                                              df_patterns, fmt_pat)
    print(f"[5] Matched {len(matches):,} patterns; running greedy assignment…")
    kept = greedy_assignment(matches)
    backfill_net_sigma(kept, ig_signed_by_pos)
    kept = [m for m in kept if abs(m.net_sigma_ig) > 1e-9]
    kept.sort(key=lambda m: -abs(m.net_sigma_ig))
    print(f"    kept {len(kept)} non-overlapping patterns "
          f"(sum |net|={sum(abs(m.net_sigma_ig) for m in kept):.3f})")

    # Residual: salient positions not claimed by any kept pattern
    all_claimed = set()
    for m in kept:
        all_claimed |= m.net_positions
    residual_positions = set()
    for _, row in df_sal.iterrows():
        key = (int(row["pos"]), row["token_str"])
        if key not in all_claimed:
            residual_positions.add(key)
    residual_sigma = sum(ig_signed_by_pos.get(p, 0.0)
                          for (p, _) in residual_positions)
    print(f"    residual salient: {len(residual_positions)} positions "
          f"(Σ={residual_sigma:+.3f})")

    # Non-salient bucket: all IG positions not in the salient set.
    # Adding this bar to the waterfall makes the bars sum to total Σ IG,
    # which by Captum's completeness axiom equals ŷ(x) − ŷ(baseline).
    nonsal_rows = df_ig[df_ig["is_salient"] == False]  # noqa: E712
    nonsalient_sigma = float(nonsal_rows["ig_signed"].sum())
    nonsalient_count = int(len(nonsal_rows))
    total_signed_ig = float(df_ig["ig_signed"].sum())
    print(f"    non-salient: {nonsalient_count} positions "
          f"(Σ={nonsalient_sigma:+.3f})")
    print(f"    total Σ IG: {total_signed_ig:+.3f}  (≈ ŷ − baseline)")

    # ── Visit times + next admission ───────────────────────────────────
    visit_times, next_adm = fetch_visit_times(pid, hadm, args.mimic_dir)
    if visit_times is not None:
        print(f"[6] Visit times: {len(visit_times)} visits over "
              f"{visit_times['days_since_v1'].iloc[-1]:.0f} days")
    if next_adm is not None:
        print(f"    next admission: {next_adm['days_post']} days post-discharge")

    # ── Waterfall: trim to top-N by |net σIG|, then ORDER FIRST → MOST RECENT ──
    idx_x = n_visits - 1
    # Top of waterfall = patterns reaching back to V1 (oldest reach).
    # Bottom = patterns at the index visit only (most recent).
    # Within same earliest-visit bucket, sort by |net σIG| descending.
    waterfall_kept = kept[:args.top_waterfall]
    extra_kept = kept[args.top_waterfall:]
    extra_sigma = sum(m.net_sigma_ig for m in extra_kept)
    extra_n = sum(len(m.net_positions) for m in extra_kept)
    if extra_n > 0:
        residual_sigma += extra_sigma
        residual_positions_count = len(residual_positions) + extra_n
    else:
        residual_positions_count = len(residual_positions)

    def _earliest_vi(m):
        if not m.matched_blocks:
            return idx_x
        return min(vi for (_, vi) in m.matched_blocks)
    def _latest_vi(m):
        if not m.matched_blocks:
            return idx_x
        return max(vi for (_, vi) in m.matched_blocks)
    # Sort ASCENDING by (earliest_vi, latest_vi): patterns at the same
    # starting visit are then ordered by reach, single-visit patterns
    # (e.g., V2: ...) appear ABOVE cross-visit ones (e.g., V2→V5: ...)
    # because a smaller latest_vi sorts earlier. Within identical reach,
    # larger |σIG| comes first.
    waterfall_kept.sort(key=lambda m: (_earliest_vi(m), _latest_vi(m),
                                          -abs(m.net_sigma_ig)))

    # ── [TOK] Per-visit token selection for Panel A ────────────────────
    # Rule: show every salient token that is CITED by any of Panel B's
    # top-N kept patterns (i.e., the token appears in the pattern's full
    # claimed_positions set, regardless of whether the greedy allocator
    # later credits the token's IG to that specific bar or to a higher-
    # priority overlapping bar). This ensures, e.g., that V1 Trazodone
    # surfaces in Panel A because it's cited by `V1→V2: Trazodone →
    # Anxiety NOS` in Panel B, even though Trazodone's IG mass got
    # marginally re-allocated under the non-overlapping rule.
    #
    # If a visit has FEWER than MIN_PER_VISIT cited tokens, top it up
    # with its highest-|IG| salient tokens to reach MIN_PER_VISIT,
    # those filler tokens still get a direction glyph (▲ / ▼) and color
    # to keep visual consistency, since the user wants direction-only
    # encoding regardless of "surfaced vs not." Per-token IG numbers
    # remain omitted (their per-token arithmetic doesn't match the
    # marginal bar values in Panel B; see Methods).
    cited_positions: set = set()
    for m in waterfall_kept:
        for (p, _t) in m.claimed_positions:
            cited_positions.add(int(p))
    MIN_PER_VISIT = 3     # top up to this many tokens per visit
    PER_VISIT_CAP = 3     # hard cap: at most this many tokens per visit.
                          # Visits that cite more than 3 Panel B tokens
                          # show only the top-3 by |IG|, keeps each
                          # visit's column visually balanced. The
                          # additional cited tokens are still visible in
                          # Panel B's pattern bars.
    visit_top_tokens = {}
    for vi, g in df_sal.groupby("visit_idx"):
        rows = g.copy()
        rows["_cited"] = rows["pos"].astype(int).isin(cited_positions)
        cited = (rows[rows["_cited"]]
                  .sort_values("ig_abs", ascending=False))
        out = [(fmt_short(r["token_str"]), float(r["ig_signed"]))
               for _, r in cited.iterrows()]
        # Top up with highest-|IG| non-cited tokens until MIN_PER_VISIT
        if len(out) < MIN_PER_VISIT:
            remaining = (rows[~rows["_cited"]]
                          .sort_values("ig_abs", ascending=False)
                          .head(MIN_PER_VISIT - len(out)))
            for _, r in remaining.iterrows():
                out.append((fmt_short(r["token_str"]),
                            float(r["ig_signed"])))
        visit_top_tokens[int(vi)] = out[:PER_VISIT_CAP]

    # ── Compose figure ─────────────────────────────────────────────────
    # Add 1 row to account for the non-salient bar in the bottom panel.
    n_water = len(waterfall_kept) + (1 if abs(residual_sigma) > 1e-9 else 0) + 1
    fig_h = max(13, 6.5 + n_water * 0.36)
    fig_w = max(15, 4 + (n_visits + 1) * 2.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Top row spans the full figure width (visit timeline has no need to
    # align with bar columns below). Bottom row keeps the 2-column
    # labels-vs-bars split so the pattern labels sit cleanly to the left
    # of their chevron bars.
    #
    # The figure has NO embedded title or legend; both live in the
    # manuscript caption to avoid duplication and free up panel space.
    outer = fig.add_gridspec(2, 2,
                              width_ratios=[1.35, 2.0],   # wider label column so
                                                          # long pattern labels
                                                          # (e.g. "V2→V5: Cancer
                                                          # pain → {Cancer anemia;
                                                          # Cancer pain}") fit
                                                          # without truncation.
                              height_ratios=[2.4, max(1.5, n_water * 0.20)],
                              hspace=0.20, wspace=0.0,
                              top=0.96, bottom=0.05, left=0.04, right=0.98)
    # Visit timeline spans the full width.
    ax_top         = fig.add_subplot(outer[0, :])
    ax_bot_labels  = fig.add_subplot(outer[1, 0])
    ax_bot         = fig.add_subplot(outer[1, 1])

    # Top: timeline
    render_timeline(ax_top, n_visits, visit_times, next_adm,
                     visit_top_tokens, y_true, y_prob, fmt_short)
    ax_top.set_title(
        "A · Patient timeline with top salient tokens per admission",
        loc="left", fontsize=14, fontweight="bold", pad=6)

    # Bottom: greedy waterfall (split: labels axes + bars axes)
    render_greedy_waterfall(ax_bot, ax_bot_labels, waterfall_kept,
                              residual_sigma, residual_positions_count,
                              fmt_pat, preserve_order=True,
                              nonsalient_sigma=nonsalient_sigma,
                              nonsalient_count=nonsalient_count,
                              total_signed_ig=total_signed_ig,
                              y_prob=y_prob)
    ax_bot_labels.set_title("B · Pattern contributions to readmission risk",
                              loc="left", fontsize=14, fontweight="bold", pad=6)

    # ── Save ───────────────────────────────────────────────────────────
    if args.no_id_suffix:
        out_path = args.output_path
    else:
        root, ext = os.path.splitext(args.output_path)
        out_path = f"{root}_{pid}{ext}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(out_path) // 1024
    print(f"[7] Saved → {out_path} ({sz} KB)")


if __name__ == "__main__":
    main()
