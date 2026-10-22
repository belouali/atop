"""Chart primitives for AToP figures.

Flat API in the style of ``shap.plots``, each function renders one chart type
into a matplotlib ``Axes`` (creates one if not provided) and returns it.

Multi-panel publication figures are composed by ``paper/jamia2026/figures/make_fig*.py``,
the figure-level layout (gridspec, suptitle, panel titles) belongs to the
caller; primitives only know about their own axes.

Available primitives (added incrementally as ``make_fig*.py`` scripts land):

  - ``ig_waterfall``, top-N IG token waterfall + "Other" aggregate
  - ``visit_block_bars``, per-visit ΣIG bars with token labels
  - ``pattern_waterfall``, matched-pattern cumulative waterfall
  - ``validation_dumbbell``, perturbation-validation dumbbell (Δŷ)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Default colors (locked decisions, see TASKS/_shared_context.md)
RISK = "#ca0020"
PROT = "#92c5de"
NEUTRAL_GRAY = "#6c6c6c"
LIGHT_GRAY = "#b5b5b5"

# Dumbbell uses different reds/blues (set during the JAMIA review).
DUMBBELL_RISK = "#c44e52"
DUMBBELL_PROT = "#4c72b0"
DUMBBELL_BASELINE = "#444444"


# ════════════════════════════════════════════════════════════════════════════
# IG / token plots
# ════════════════════════════════════════════════════════════════════════════


def ig_waterfall(
    *,
    df_top: pd.DataFrame,
    other_value: float,
    n_other_tokens: int,
    label_formatter: Callable[[str, int], str],
    y_prob: float,
    ax: Optional[plt.Axes] = None,
    risk_color: str = RISK,
    prot_color: str = PROT,
    label_max_len: int = 42,
    tick_color_fn: Optional[Callable[[str], str]] = None,
) -> plt.Axes:
    """Render a SHAP-style IG waterfall: cumulative walk from baseline E[f(x)] to f(x).

    Args
    ----
    df_top:           rows for the top-N tokens, sorted by signed IG ascending
                      (most negative first → most positive last). Required cols:
                      ``token_str``, ``ig_signed``.
    other_value:      summed IG of the tokens NOT shown ("Other" aggregate).
    n_other_tokens:   how many tokens contributed to ``other_value``.
    label_formatter:  callable ``(token_str, max_len)`` → display string.
    y_prob:           model prediction f(x), shown as the right-edge annotation.
    ax:               axes to draw on; created if ``None``.

    Returns the axes for further customization. The "A · Integrated Gradients
    waterfall" panel title is set by the caller.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    n_entries = len(df_top) + 1  # +1 for 'Other' at the bottom
    cum = 0.0
    positions = []
    for i in range(n_entries):
        if i == 0:
            val = other_value
            label = f"Other ({n_other_tokens} tokens)"
        else:
            row = df_top.iloc[i - 1]
            val = float(row["ig_signed"])
            label = label_formatter(row["token_str"], label_max_len)
        positions.append((cum, val, label))
        cum += val

    shown_ig_abs = abs(sum(v for _, v, _ in positions if v != other_value))
    if shown_ig_abs == 0:
        shown_ig_abs = max(abs(cum), 1e-6)

    for i, (start, val, _label) in enumerate(positions):
        color = risk_color if val > 0 else prot_color
        ax.barh(i, val, left=start, color=color, edgecolor="white",
                linewidth=0.8, height=0.72)
        end_x = start + val
        ha = "left" if val >= 0 else "right"
        off = shown_ig_abs * 0.008 if val >= 0 else -shown_ig_abs * 0.008
        ax.text(end_x + off, i, f"{val:+.3f}",
                ha=ha, va="center", fontsize=9.5, fontweight="bold", color="#222222")

    y = np.arange(n_entries)
    ax.set_yticks(y)
    ax.set_yticklabels([pos[2] for pos in positions], fontsize=10)

    # Per-tick stream coloring (C=dx blue / P=proc orange / D=drug green).
    # ``tick_color_fn`` takes the raw token_str and returns a hex color.
    # The first row is always the "Other" aggregate, which gets gray.
    if tick_color_fn is not None:
        raw_tokens = ["OTHER"] + [df_top.iloc[i]["token_str"] for i in range(len(df_top))]
        for tick, raw in zip(ax.get_yticklabels(), raw_tokens):
            tick.set_color("#999999" if raw == "OTHER" else tick_color_fn(raw))

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(cum, color="black", linewidth=0.8, linestyle=":")

    ax.annotate(f"f(x) = {y_prob:.3f}", xy=(cum, n_entries - 0.7),
                xytext=(cum, n_entries - 0.2), ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.annotate("E[f(x)]  (baseline)", xy=(0, -0.5),
                xytext=(0, -1.0), ha="center", va="top",
                fontsize=10, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))

    ax.set_xlabel("Signed IG contribution", fontsize=11)
    ax.set_ylim(-1.5, n_entries - 0.3)
    xmin = min(p[0] for p in positions) - shown_ig_abs * 0.15
    xmax = max(p[0] + p[1] for p in positions) + shown_ig_abs * 0.15
    ax.set_xlim(xmin, xmax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(handles=[
        Patch(facecolor=risk_color, label="Risk (IG > 0)"),
        Patch(facecolor=prot_color, label="Protective (IG < 0)"),
    ], loc="lower right", fontsize=9, framealpha=0.95)

    return ax


def visit_block_bars(
    *,
    visit_sums: np.ndarray,
    visit_top_tokens: Dict[int, List[Tuple[str, float]]],
    visit_nonsal_count: Dict[int, int],
    visit_salient_count: Dict[int, int],
    visit_times: Optional[pd.DataFrame],
    code_formatter: Callable[[str], str],
    ax: Optional[plt.Axes] = None,
    risk_color: str = RISK,
    prot_color: str = PROT,
    neutral_gray: str = NEUTRAL_GRAY,
    light_gray: str = LIGHT_GRAY,
    cumulative_axis: bool = True,
) -> plt.Axes:
    """Per-visit ΣIG bars with top-K salient tokens labelled and non-salient
    counts below the x-axis. Optional secondary right axis with cumulative ΣIG;
    optional secondary top axis with days since first visit (if visit_times).

    Args
    ----
    visit_sums:           length n_visits, sum of IG per visit.
    visit_top_tokens:     {visit_idx: [(token_str, ig_signed), ...]}
    visit_nonsal_count:   {visit_idx: int}, non-salient tokens per visit.
    visit_salient_count:  {visit_idx: int}, total salient tokens (used for
                          "+N more" annotation if more than displayed).
    visit_times:          optional DataFrame with ``visit_idx`` and
                          ``days_since_v1`` for the top axis.
    code_formatter:       callable ``(token_str)`` → short display code.
    """
    n_visits = len(visit_sums)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    visit_colors = [risk_color if v > 0 else prot_color for v in visit_sums]
    x_v = np.arange(n_visits)
    ax.bar(x_v, visit_sums, color=visit_colors, edgecolor="black",
           linewidth=0.5, width=0.75)

    cum_ig = np.cumsum(visit_sums)
    if cumulative_axis:
        ax_r = ax.twinx()
        ax_r.plot(x_v, cum_ig, color="#333333", linestyle="--", linewidth=1.6,
                  marker="o", markersize=5, label="Cumulative ΣIG")
        ax_r.set_ylabel("Cumulative ΣIG", fontsize=10, color="#333333")
        ax_r.tick_params(axis="y", colors="#333333", labelsize=9)

    y_abs = max(max(abs(visit_sums)), 0.001)
    ax.set_ylim(-y_abs * 2.0, y_abs * 2.2)
    if cumulative_axis:
        c_abs = max(max(abs(cum_ig)), 0.001)
        ax_r.set_ylim(-c_abs * 1.1, c_abs * 1.25)

    rotate_deg = 0 if n_visits <= 10 else 25
    label_ha = "center" if rotate_deg == 0 else "right"
    for vi in range(n_visits):
        toks = visit_top_tokens.get(vi, [])
        y_base = visit_sums[vi]
        if y_base >= 0:
            y_label_start = y_base + y_abs * 0.16
            stack_dir = +1
        else:
            y_label_start = y_base - y_abs * 0.16
            stack_dir = -1
        for j, (tok_str, sig) in enumerate(toks):
            label = code_formatter(tok_str)
            if len(label) > 20:
                label = label[:18] + "…"
            color = risk_color if sig > 0 else prot_color
            ax.text(vi, y_label_start + stack_dir * j * (y_abs * 0.30),
                    label, ha=label_ha, va="bottom" if stack_dir > 0 else "top",
                    fontsize=8, color=color, rotation=rotate_deg)
        extra_salient = visit_salient_count.get(vi, 0) - len(toks)
        if extra_salient > 0:
            y_extra = y_label_start + stack_dir * len(toks) * (y_abs * 0.30)
            ax.text(vi, y_extra, f"+{extra_salient} more",
                    ha=label_ha, va="bottom" if stack_dir > 0 else "top",
                    fontsize=7, color=neutral_gray, rotation=rotate_deg,
                    fontstyle="italic")
        n_ns = visit_nonsal_count.get(vi, 0)
        if n_ns > 0:
            y_below = -y_abs * 1.65
            ax.text(vi, y_below, f"+{n_ns}",
                    ha="center", va="top", fontsize=7.5,
                    color=light_gray, fontstyle="italic")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x_v)
    ax.set_xticklabels([f"V{i+1}" for i in range(n_visits)], fontsize=10)
    ax.set_xlabel("Visit (chronological)", fontsize=11)
    ax.set_ylabel("Visit ΣIG", fontsize=11)

    if visit_times is not None and len(visit_times) == n_visits:
        days = visit_times["days_since_v1"].values
        ax_t = ax.twiny()
        ax_t.set_xlim(ax.get_xlim())
        ax_t.set_xticks(x_v)
        ax_t.set_xticklabels([f"day {int(d)}" for d in days],
                             fontsize=8.5, color="#444444")
        ax_t.set_xlabel("Days since first visit", fontsize=10, color="#444444", labelpad=6)
        ax_t.spines["top"].set_color("#888888")
        for side in ("right", "left"):
            ax_t.spines[side].set_visible(False)
        ax_t.tick_params(axis="x", colors="#444444", length=3, pad=2)

    for side in ("top",):
        ax.spines[side].set_visible(False)
        if cumulative_axis:
            ax_r.spines[side].set_visible(False)

    handles = [
        Patch(facecolor=risk_color, label="Risk visit"),
        Patch(facecolor=prot_color, label="Protective visit"),
    ]
    if cumulative_axis:
        handles.append(Line2D([0], [0], color="#333333", linestyle="--", marker="o",
                              label="Cumulative ΣIG (right axis)"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    return ax


# ════════════════════════════════════════════════════════════════════════════
# Bar charts (signed IG / ΣIG), used by Fig 3 panels A/B/C + Fig S1
# ════════════════════════════════════════════════════════════════════════════


def signed_ig_bars(
    *,
    values: Sequence[float],
    labels: Sequence[str],
    ax: Optional[plt.Axes] = None,
    risk_color: str = DUMBBELL_RISK,
    prot_color: str = DUMBBELL_PROT,
    bar_height: float = 0.5,
    fontsize: int = 13,
    show_value_annotations: bool = False,
    tick_colors: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Horizontal signed-IG bar chart, colored by sign.

    Re-used across:
      * Fig 3 Panel A, population-weighted E[IG]
      * Fig 3 Panel B, carrier-conditional E[IG | present]
      * Fig 3 Panel C, pattern carrier-conditional E[ΣIG | present]
      * Fig S1       , reversed-order ablation deltas

    Caller is responsible for sorting (typically ``|values|`` ascending so the
    largest magnitude is at the top of the chart) and for setting the title /
    xlabel via the returned axes.

    Args
    ----
    values:  signed IG / ΣIG values in display order (bottom → top).
    labels:  display labels (same order).
    show_value_annotations: if True, write the numeric value at each bar end.
    tick_colors: optional per-label color (same length as labels). Used to
        color-code y-tick labels by stream (C/P/D) so readers can see at a
        glance whether each row is a diagnosis, procedure, or drug. If
        omitted, tick labels stay default black.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(6, len(values) * 0.45)))

    n = len(values)
    colors = [risk_color if v > 0 else prot_color for v in values]
    ax.barh(range(n), values, height=bar_height, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)

    # Per-tick coloring for stream identity (C=dx, P=proc, D=drug).
    if tick_colors is not None:
        if len(tick_colors) != n:
            raise ValueError(
                f"tick_colors length ({len(tick_colors)}) must match labels ({n})")
        for tick, c in zip(ax.get_yticklabels(), tick_colors):
            tick.set_color(c)

    if show_value_annotations:
        x_abs_max = max(abs(v) for v in values) if values else 1.0
        pad = x_abs_max * 0.02
        for i, v in enumerate(values):
            ha = "left" if v >= 0 else "right"
            off = pad if v >= 0 else -pad
            ax.text(v + off, i, f"{v:+.3f}",
                    ha=ha, va="center", fontsize=fontsize - 2, color="#222222")

    return ax


# ════════════════════════════════════════════════════════════════════════════
# Pattern plots
# ════════════════════════════════════════════════════════════════════════════


def pattern_waterfall(
    *,
    pattern_values: List[float],
    pattern_labels: List[str],
    other_value: float,
    n_other_patterns: int,
    cum_pct: float,
    n_total_multi_token: int,
    ax: Optional[plt.Axes] = None,
    risk_color: str = RISK,
    prot_color: str = PROT,
    label_max_len: int = 75,
    tick_colors: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Matched-pattern cumulative waterfall (Fig 2 Panel C grammar).

    Mirrors ``ig_waterfall``: bars stacked vertically, "Other" at the bottom,
    individual patterns ordered by signed ΣIG ascending. Each bar starts where
    the previous ended; running total is annotated at the right edge.

    Args
    ----
    pattern_values:       list of signed ΣIG values for the top patterns,
                          ordered ascending (most negative first).
    pattern_labels:       same order; pre-formatted display labels (caller
                          handles truncation if desired).
    other_value:          summed ΣIG of the patterns NOT shown.
    n_other_patterns:     count of patterns aggregated into ``other_value``.
    cum_pct:              percent of total |ΣIG| accounted for by the top patterns
                          (used in the panel title, caller sets the title).
    n_total_multi_token:  total multi-token matched patterns (also for title).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    if not pattern_values:
        ax.text(0.5, 0.5, "No patterns matched this patient.",
                ha="center", va="center", fontsize=12)
        return ax

    # Truncate labels (script can pre-truncate too; this is a safety net)
    labels_safe = []
    for lbl in pattern_labels:
        if len(lbl) > label_max_len:
            labels_safe.append(lbl[: label_max_len - 1] + "…")
        else:
            labels_safe.append(lbl)

    # Bottom row = "Other" aggregate
    vals_c = [other_value] + list(pattern_values)
    labels_c = [f"Other ({n_other_patterns} patterns)"] + labels_safe

    # Cumulative walk
    cum_pos = 0.0
    starts = []
    for v in vals_c:
        starts.append(cum_pos)
        cum_pos += v
    total_c = cum_pos

    y_c = np.arange(len(vals_c))
    colors_c = [risk_color if v > 0 else prot_color for v in vals_c]
    ax.barh(y_c, vals_c, left=starts, color=colors_c,
            edgecolor="white", linewidth=0.8, height=0.72)
    ax.set_yticks(y_c)
    ax.set_yticklabels(labels_c, fontsize=9.5)

    # Per-tick stream colors. ``tick_colors`` length must equal len(pattern_values);
    # the leading "Other" row is always gray.
    if tick_colors is not None:
        if len(tick_colors) != len(pattern_values):
            raise ValueError(
                f"tick_colors length ({len(tick_colors)}) must match "
                f"pattern_values ({len(pattern_values)})")
        full_colors = ["#999999"] + list(tick_colors)
        for tick, c in zip(ax.get_yticklabels(), full_colors):
            tick.set_color(c)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(total_c, color="black", linewidth=0.8, linestyle=":")

    pad = max(abs(v) for v in vals_c) * 0.015 if any(vals_c) else 0.01
    for i, (start, v) in enumerate(zip(starts, vals_c)):
        if v == 0:
            continue
        end_x = start + v
        ha = "left" if v >= 0 else "right"
        off = pad if v >= 0 else -pad
        ax.text(end_x + off, i, f"{v:+.3f}",
                ha=ha, va="center", fontsize=9.5, fontweight="bold", color="#222222")

    ax.annotate(
        f"Σ (top {len(pattern_values)} + other) = {total_c:+.3f}",
        xy=(total_c, len(vals_c) - 0.6), xytext=(total_c, len(vals_c) - 0.1),
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )
    ax.annotate(
        "baseline (0)",
        xy=(0, -0.5), xytext=(0, -1.0),
        ha="center", va="top", fontsize=10, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
    )

    ax.set_xlabel(
        "Cumulative Σ IG from matched patterns\n← protective            risk →",
        fontsize=11,
    )
    bounds = [0.0, total_c] + starts + [s + v for s, v in zip(starts, vals_c)]
    vpad = max(abs(v) for v in vals_c) * 0.15 if any(vals_c) else 0.05
    ax.set_xlim(min(bounds) - vpad, max(bounds) + vpad)
    ax.set_ylim(-1.5, len(vals_c) - 0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return ax


# ════════════════════════════════════════════════════════════════════════════
# Comparison plots, generic, agnostic to what the values represent
# ════════════════════════════════════════════════════════════════════════════


def dumbbell(
    *,
    baseline: Sequence[float],
    after: Sequence[float],
    labels: Sequence[str],
    ax: Optional[plt.Axes] = None,
    increase_color: str = DUMBBELL_RISK,
    decrease_color: str = DUMBBELL_PROT,
    baseline_color: str = DUMBBELL_BASELINE,
    fontsize: int = 13,
    fig_width: float = 14,
    bar_squeeze: float = 0.5,
    annotate_change: bool = True,
    annotate_decimals: int = 3,
    epsilon: float = 0.0005,
    legend_labels: Optional[Sequence[str]] = None,
    show_baseline: bool = True,
    show_legend: bool = True,
    suppress_tick_labels: bool = False,
) -> plt.Axes:
    """Generic horizontal dumbbell plot, before vs after for a set of items.

    For each row:
      - Square at ``baseline[i]``
      - Circle at ``after[i]``
      - Arrow connecting them, colored by direction (increase / decrease)

    The plot is agnostic about what the values represent. The caller (e.g.
    ``paper/jamia2026/figures/make_fig4.py``) decides what "baseline" and "after" mean,
    perturbation Δŷ, intervention pre/post, IG-rank shifts, etc., and supplies
    the title, x-label, and legend wording from outside.

    Args
    ----
    baseline:        N values, the "before" position (drawn as squares).
    after:           N values, the "after" position (drawn as circles).
    labels:          N display labels (y-axis ticks).
    increase_color:  arrow color when ``after > baseline``.
    decrease_color:  arrow color when ``after < baseline``.
    epsilon:         changes smaller than this in absolute value get a
                     neutral-gray arrow and no text annotation.
    legend_labels:   optional 3-tuple ``(square, increase, decrease)`` for the
                     standard legend; pass ``None`` to skip the legend
                     (the caller adds their own with domain-specific wording).

    Note: ``fig_width`` and ``bar_squeeze`` are honored only when this function
    creates its own figure (``ax is None``); when composing, the caller controls
    figure size.

    Returns the axes for further customization (titles, x-labels, additional
    legends).
    """
    N = len(baseline)
    if N == 0:
        raise ValueError("Empty input, nothing to plot")
    if not (len(after) == N and len(labels) == N):
        raise ValueError(f"Length mismatch: baseline={N}, after={len(after)}, labels={len(labels)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(fig_width, max(6, N * 0.55)))
        owns_fig = True
    else:
        owns_fig = False

    fs_xlabel = fontsize - 1
    fs_legend = fontsize - 1

    y = np.arange(N)
    for i in range(N):
        x_start = baseline[i]
        x_end = after[i]
        change = x_end - x_start

        if abs(change) < epsilon:
            color = "#999999"
        elif change > 0:
            color = increase_color
        else:
            color = decrease_color

        # Square (baseline)
        ax.plot(x_start, y[i], "s", color=baseline_color, markersize=9, zorder=3)

        # Arrow
        if abs(change) >= epsilon:
            ax.annotate(
                "",
                xy=(x_end, y[i]), xytext=(x_start, y[i]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                shrinkA=5, shrinkB=2),
            )

        # Circle (after)
        ax.plot(x_end, y[i], "o", color=color, markersize=9, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

        # Magnitude annotation
        if annotate_change and abs(change) >= epsilon:
            sign = "+" if change > 0 else ""
            ax.text((x_start + x_end) / 2, y[i] + 0.25,
                    f"{sign}{change:.3f}", fontsize=fontsize - 6, color=color,
                    ha="center", va="bottom", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize, color="#1f2937")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=fs_xlabel - 1)

    if legend_labels is not None:
        sq_lbl, inc_lbl, dec_lbl = legend_labels
        legend_handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=baseline_color,
                   markersize=9, label=sq_lbl),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=increase_color,
                   markersize=9, label=inc_lbl),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=decrease_color,
                   markersize=9, label=dec_lbl),
        ]
        ax.legend(
            handles=legend_handles, fontsize=fs_legend,
            loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3, frameon=False,
        )

    if owns_fig:
        ax.figure.tight_layout()
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.05, pos.width * bar_squeeze,
                         pos.height * 0.92])

    return ax
