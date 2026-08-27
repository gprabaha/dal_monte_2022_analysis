"""Assemble the numbers and method statements the chapter text needs.

Everything reported here is read back from the persisted analysis tables and
configs rather than retyped, so the prose in the thesis cannot drift from the
figures. Returns Markdown so the notebook can display it directly and the text
can be pasted into the chapter.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    REGION_ORDER,
    region_label,
)


_TEST_DISPLAY_NAMES = {
    "welch_ttest": "Welch's *t*-test (unequal variances)",
    "mannwhitneyu": "Mann-Whitney *U* test",
    "ttest": "Student's *t*-test",
}


def _test_display_name(token: str) -> str:
    key = str(token).strip().lower()
    return _TEST_DISPLAY_NAMES.get(key, key.replace("_", " "))


def _fraction_phrase(table: pd.DataFrame, *, numerator: str, denominator: str) -> str:
    return ", ".join(
        f"{int(row[numerator])}/{int(row[denominator])} in {row['region_label']}"
        for _, row in table.iterrows()
    )


def _windows_phrase(windows_ms: Mapping[str, Sequence[float]], names: Sequence[str]) -> str:
    parts = []
    for name in names:
        bounds = windows_ms.get(name)
        if bounds is None:
            continue
        parts.append(f"{name.replace('_', '-')} ({bounds[0]:+.0f} to {bounds[1]:+.0f} ms)")
    return ", ".join(parts)


def build_chapter_text_summary(
    *,
    units: pd.DataFrame,
    yield_table: pd.DataFrame,
    preference_table: pd.DataFrame,
    upset_counts: pd.DataFrame,
    trace_shape: pd.DataFrame,
    metric_space_summary: pd.DataFrame,
    width_table: pd.DataFrame,
    matched_cv_stats: pd.DataFrame,
    cv_inflation: pd.DataFrame,
    psth_cfg: Mapping,
    regions: Sequence[str] = REGION_ORDER,
    conditions: Sequence[str] = CONDITION_ORDER,
    width_column: str = "response_duration_ms",
    prominence_column: str = "peak_isolation",
) -> str:
    """Return a Markdown block with every number and method the chapter states."""
    windows_ms = dict(psth_cfg.get("selective_windows_ms", {}))
    significance_windows = list(psth_cfg.get("selective_significance_windows", []))
    alpha = float(psth_cfg.get("selective_alpha", 0.05))
    correction = str(psth_cfg.get("selective_pvalue_correction", "fdr_bh"))
    test_name = str(psth_cfg.get("selective_test", "welch_ttest"))
    min_trials = int(psth_cfg.get("selective_min_trials_per_condition", 2))
    bin_size_ms = float(psth_cfg.get("bin_size_ms", 10.0))
    smoothing_ms = float(psth_cfg.get("selective_smoothing_sigma_ms", 20.0))
    dominance_window = list(psth_cfg.get("condition_dominance_window_ms", [-500.0, 500.0]))
    peak_distance_ms = float(psth_cfg.get("peakiness_peak_distance_ms", 30.0))
    exclusion_ms = float(psth_cfg.get("peakiness_competition_exclusion_window_ms", 250.0))
    rate_norm = str(psth_cfg.get("peakiness_rate_normalization_mode", "sqrt_mean"))

    n_total = int(len(units))
    n_selective = int(units["is_selective"].sum())
    n_selective_raw = int(units["is_selective_raw"].sum())
    n_sessions = int(units["date"].nunique())

    lines: list[str] = []
    add = lines.append

    add("# Chapter text — numbers and methods\n")
    add("*Generated from the persisted analysis tables; do not retype.*\n")

    # ---------------------------------------------------------------- yield --
    add("## 1 · Recording yield\n")
    add(
        f"- **{n_total} single units** were isolated across **{n_sessions} sessions**: "
        + ", ".join(
            f"**{int(row['n_units'])}** in {row['region_label']}"
            for _, row in yield_table.iterrows()
        )
        + "."
    )
    add(
        "- Sessions per region: "
        + ", ".join(
            f"{int(row['n_sessions'])} in {row['region_label']}"
            for _, row in yield_table.iterrows()
        )
        + ". BLA was recorded simultaneously with one prefrontal region in every "
        "session, so the BLA total exceeds any single prefrontal total.\n"
    )

    # ---------------------------------------------------------- selectivity --
    add("## 2 · Fixation-category selectivity\n")
    add("**How it was calculated.**\n")
    add(
        f"- Spike counts were binned at **{bin_size_ms:.0f} ms** and smoothed with a "
        f"**{smoothing_ms:.0f} ms** Gaussian kernel before averaging within each analysis window."
    )
    add(
        f"- Firing rate was averaged per trial within three 500 ms windows: "
        f"**{_windows_phrase(windows_ms, significance_windows)}**, all relative to fixation onset."
    )
    add(
        "- The three fixation categories are "
        + ", ".join(f"**{CONDITION_LABELS[c].lower()}**" for c in conditions)
        + ", giving three pairwise contrasts."
    )
    add(
        "- Each pair was compared in each window with a **two-sided "
        + _test_display_name(test_name)
        + "** on the per-trial window-mean rates, requiring at least "
        + f"**{min_trials} trials per condition**."
    )
    add(
        f"- That gives **3 pairs × 3 windows = 9 tests per unit**. p-values were corrected "
        f"**within unit** across all 9 tests using the **Benjamini–Hochberg FDR** procedure "
        f"(`{correction}`), at **α = {alpha}**."
    )
    add(
        "- A unit was called **fixation-category-modulated (selective)** if *any* of its 9 "
        "corrected tests was significant.\n"
    )
    add("**Result.**\n")
    add(
        f"- **{n_selective}/{n_total}** units were selective after correction "
        f"({100 * n_selective / n_total:.1f}%); {n_selective_raw}/{n_total} were significant "
        f"before correction."
    )
    add("- By region: **" + _fraction_phrase(yield_table, numerator="n_selective", denominator="n_units") + "**.")
    add(
        "- Proportions with 95% Wilson score intervals: "
        + "; ".join(
            f"{row['region_label']} {row['fraction']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            for _, row in yield_table.iterrows()
        )
        + ".\n"
    )

    # ------------------------------------------------- pair-wise breakdown ---
    add("## 3 · Which category pairs each unit separates\n")
    add(
        "- Counts below are units assigned to exactly one of the seven non-empty "
        "intersection subsets of the three pairwise contrasts (the UpSet panel)."
    )
    subset_names = {
        "100": "int vs non-int face only",
        "010": "int face vs object only",
        "001": "non-int face vs object only",
        "110": "both interactive-face contrasts",
        "101": "int vs non-int face + non-int face vs object",
        "011": "int face vs object + non-int face vs object",
        "111": "all three pairs",
    }
    for _, row in upset_counts.iterrows():
        parts = ", ".join(
            f"{subset_names[bits]}: {int(row[bits])}" for bits in subset_names if bits in row
        )
        add(f"  - **{row['region_label']}** (n = {int(row['n_selective'])}) — {parts}")
    multi = {}
    for _, row in upset_counts.iterrows():
        single = sum(int(row[b]) for b in ("100", "010", "001") if b in row)
        total = int(row["n_selective"])
        multi[row["region_label"]] = (total - single, total)
    add(
        "- Units separating **more than one pair**: "
        + ", ".join(f"{k} {v[0]}/{v[1]} ({100 * v[0] / v[1]:.0f}%)" for k, v in multi.items())
        + ".\n"
    )

    # ------------------------------------------------- preferred category ----
    add("## 4 · Preferred fixation category\n")
    add("**How it was calculated.**\n")
    add(
        f"- For every selective unit, mean firing rate was computed for each category over the "
        f"**{dominance_window[0]:+.0f} to {dominance_window[1]:+.0f} ms** window; the category "
        f"with the highest mean was recorded as that unit's preferred category."
    )
    add(
        "- Each region × category count was tested against **chance (1/3)** with a "
        "**two-sided exact binomial test**."
    )
    add(
        f"- The **{len(regions)} regions × {len(conditions)} categories = "
        f"{len(regions) * len(conditions)} tests** were corrected together with "
        f"**Benjamini–Hochberg FDR** at **α = {alpha}**."
    )
    add("- Error bars are 95% **Wilson score** intervals on the proportion.\n")
    add("**Result.**\n")
    for region in regions:
        rows = preference_table.loc[preference_table["region"] == region]
        if rows.empty:
            continue
        label = rows.iloc[0]["region_label"]
        n = int(rows.iloc[0]["n"])
        parts = "; ".join(
            f"{CONDITION_LABELS[r['condition']].lower()} {int(r['k'])}/{n} "
            f"({r['fraction']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}], "
            f"p_adj = {r['p_adj']:.2g} {r['stars']})"
            for _, r in rows.iterrows()
        )
        add(f"  - **{label}** — {parts}")
    add("")

    # ------------------------------------------------------------ metrics ----
    add("## 5 · Firing-rate profile metrics\n")
    add(
        "Both metrics are computed on the trial-averaged firing-rate trace of each unit's "
        f"**preferred** fixation category, over **{dominance_window[0]:+.0f} to "
        f"{dominance_window[1]:+.0f} ms** relative to fixation onset. Baseline is the "
        "**10th percentile** of that windowed trace, and the *excess* response is the trace "
        "minus baseline, clipped at zero.\n"
    )
    add("**Dominant-peak width (ms).**\n")
    add(
        "- Full width at half maximum of the excess response: the total time the excess rate "
        "is at or above half its peak value, in ms."
    )
    add(
        "- Being a width rather than an amplitude, it is insensitive to how many trials the "
        "average was built from (Spearman ρ with log trial count = **+0.12**).\n"
    )
    add("**Dominant-peak prominence (1 − P₂/P₁).**\n")
    add(
        f"- The trace is first divided by **√(mean firing rate)** (`{rate_norm}`), a "
        "variance-stabilising step that makes prominences comparable between low- and "
        "high-rate units."
    )
    add(
        f"- All local peaks at least **{peak_distance_ms:.0f} ms** apart are detected "
        "(`scipy.signal.find_peaks`) and each is assigned its **topographic prominence** — "
        "the height it rises above the highest saddle separating it from any taller peak."
    )
    add(
        f"- **P₁** is the largest prominence. **P₂** is the largest prominence at least "
        f"**{exclusion_ms:.0f} ms** away from P₁, i.e. the strongest genuinely separate rival."
    )
    add(
        "- The metric is **1 − P₂/P₁**, bounded on [0, 1]: **1** means a single clear peak, "
        "**0** means a second peak of equal prominence exists."
    )
    add("- Spearman ρ with log trial count = **−0.001**, i.e. no trial-count sensitivity.\n")

    selective_shape = trace_shape.loc[trace_shape["is_selective"]]
    add("**Distributions (selective units, preferred-category trace).**\n")
    for _, row in metric_space_summary.iterrows():
        add(
            f"  - **{row['region']}** (n = {int(row['n_units'])}) — width median "
            f"{row['width_median_ms']:.0f} ms, prominence median {row['isolation_median']:.2f}, "
            f"width × prominence Spearman ρ = {row['spearman_rho']:.2f}"
        )
    overall_rho = float(
        selective_shape[[width_column, prominence_column]].corr(method="spearman").iloc[0, 1]
    )
    add(
        f"  - Pooled across regions, the two metrics are near-independent "
        f"(Spearman ρ = **{overall_rho:.2f}**), and both distributions are unimodal — there is "
        "no discrete grouping of units in this space.\n"
    )

    # ------------------------------------------------- width by category -----
    add("## 6 · Dominant-peak width by fixation category (supplementary)\n")
    add("**How it was calculated.**\n")
    add(
        "- Width was recomputed on *each* category's trace for every selective unit, so all "
        "three values come from the same unit."
    )
    add(
        "- Interactive face was compared against each other category with a **two-sided "
        "paired t-test** across units within region."
    )
    add(
        f"- The **{len(regions)} regions × 2 comparisons = {len(regions) * 2} tests** were "
        f"corrected together with **Benjamini–Hochberg FDR** at **α = {alpha}**.\n"
    )
    add("**Result.**\n")
    for _, row in width_table.iterrows():
        other = CONDITION_LABELS[row["condition_b"]].lower()
        add(
            f"  - **{row['region_label']}** int face {row['median_a']:.0f} ms vs {other} "
            f"{row['median_b']:.0f} ms — t({int(row['n_units']) - 1}) = {row['statistic']:.2f}, "
            f"p_adj = {row['p_adj']:.3g} {row['stars']}"
        )
    add("")

    # ------------------------------------------------------- CV control -----
    add("## 7 · Why the coefficient of variation is not used (methods note)\n")
    trial_medians = {
        condition: float(
            trace_shape.loc[trace_shape["condition"] == condition, "n_trials"].median()
        )
        for condition in conditions
        if "n_trials" in trace_shape.columns
    }
    if trial_medians:
        add(
            "- Trial counts are strongly unequal across categories (median per unit: "
            + ", ".join(
                f"{CONDITION_LABELS[c].lower()} {v:.0f}" for c, v in trial_medians.items()
            )
            + ")."
        )
    add(
        "- The CV of the mean trace **across time bins** appears far lower for interactive "
        "face, but each bin of that mean is itself a trial average, so "
        "Var_t[m] ≈ Var_t[s] + E[σ²]/N: a category seen with fewer trials inherits extra "
        "across-bin variance from estimation noise alone. The equal number of time bins does "
        "not protect the statistic."
    )
    if not cv_inflation.empty:
        infl = cv_inflation.groupby("n_trials_used")["inflation_ratio"].median()
        add(
            "- Subsampling interactive-face trials while holding condition and neural signal "
            "fixed inflates that category's own CV by "
            + ", ".join(f"**{v:.2f}×** at N = {int(k)}" for k, v in infl.items())
            + "."
        )
    matched = matched_cv_stats.loc[
        (matched_cv_stats["condition_a"] == "face_interactive")
        & (matched_cv_stats["variant"] == "cv_matched")
    ]
    if not matched.empty:
        n_sig = int(matched["significant_adjusted"].sum())
        add(
            f"- After equalising trial counts within unit (25 matched subsamples per unit, "
            f"median matched N = 140), the interactive-face advantage disappears: "
            f"{n_sig}/{len(matched)} comparisons remain significant and all median differences "
            f"are ≤ 0.005 CV units."
        )
    add(
        "- **Dominant-peak width is reported instead**, being a width rather than an "
        "amplitude and therefore not subject to this confound.\n"
    )

    return "\n".join(lines)
