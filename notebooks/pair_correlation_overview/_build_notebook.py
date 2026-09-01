"""Author the combined noise + signal correlation notebook."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

FILENAME = "pair_correlation_overview.ipynb"
TITLE = "Noise and signal correlation in simultaneously recorded selective pairs"

SETUP = '''
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = next(parent for parent in Path.cwd().parents if (parent / "src").exists())
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.ephys.analysis import fixation_pair_spike_coordination as psc
from dal_monte_2022_analysis.ephys.analysis import fixation_signal_correlation as sc
from dal_monte_2022_analysis.ephys.plotting import fixation_pair_correlation_overview as viz

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

CFG_PATH = str(repo_root / "configs" / "dataset.yaml")
cfg = load_config(CFG_PATH)
FIGURE_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "overview"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figs = viz.PairOverviewPlotSettings(output_dir=FIGURE_DIR)
SUMMARY_DIR = build_analysis_output_dir(cfg, psc.DEFAULT_OUTPUT_SUBDIR) / "summary"
print("figures ->", FIGURE_DIR)
'''

SCHEMATIC = '''
fig, paths = viz.plot_method_schematic(figs)
display(Image(filename=str(paths["png"])))
'''

LOAD = '''
# --- noise correlation: per-fixation spike trains, circular-shift null -------
noise = psc.load_pair_coordination(CFG_PATH)[0]
noise, dropped = psc.drop_zero_lag_artifact_dates(noise)
noise = noise.loc[noise["both_selective"] & noise["same_region"]].copy()
noise_traces = pd.read_pickle(SUMMARY_DIR / "traces_by_region_selective.pkl")
print(f"noise: {len(noise):,} pair-conditions, both units FDR-selective, "
      f"within region   (artifact dates removed: {dropped})")

# --- signal correlation: condition-averaged timelines, cross-session null ----
signal_settings = sc.SignalCorrelationSettings(cfg_path=CFG_PATH)
units, timeline = sc.load_condition_timelines(signal_settings)
signal, signal_lags = sc.build_pair_correlations(units, timeline, signal_settings)
signal_within = signal.loc[signal["scope"] == "within_region"]
print(f"signal: {len(signal_within):,} within-region pairs from {len(units)} selective units")

display(
    noise.groupby("region_pair", observed=True)
    .agg(noise_pairs=("pair_key", "size"))
    .join(
        signal_within.groupby("region_pair", observed=True)
        .size().rename("signal_pairs").to_frame()
    )
)
'''

NOISE_ABOVE = '''
fig, paths = viz.plot_noise_above_null(noise_traces, figs, condition="face_interactive")
display(Image(filename=str(paths["png"])))
'''

NOISE_EXCESS = '''
noise_contrasts = psc.compare_conditions(
    noise, metric="circular_shift_peak_pm2ms_matched", group_columns=("region_pair",)
)
fig, paths = viz.plot_excess_by_condition(
    noise_traces, figs, contrasts=noise_contrasts,
    ylabel="Observed − null\\n(coincidences per fixation)",
    title="Fixation types do not differ — noise correlation",
    stem="fig03_noise_excess",
)
display(Image(filename=str(paths["png"])))

display(
    noise_contrasts.loc[
        :, ["region_pair", "condition_a", "condition_b", "n_pairs",
            "mean_difference", "effect_size_rank_biserial", "p_value_corrected", "significant"]
    ].round(5)
)
print(f"largest |effect size| across all contrasts: "
      f"{noise_contrasts['effect_size_rank_biserial'].abs().max():.3f}")
print(f"contrasts reaching significance: {int(noise_contrasts['significant'].sum())} "
      f"of {len(noise_contrasts)}")
'''

SIGNAL_TRACES = '''
signal_traces = {
    "lags_ms": signal_lags,
    "traces": sc.build_group_traces(signal, signal_settings),
}
fig, paths = viz.plot_excess_by_condition(
    signal_traces, figs, max_lag_ms=250.0,
    ylabel="Observed − null\\n(correlation)",
    title="Signal correlation, null-corrected",
    stem="fig04_signal_excess",
)
display(Image(filename=str(paths["png"])))
'''

SIGNAL_BANDS = '''
lag_summary = sc.summarize_lag_measures(signal, signal_settings)
fig, paths = viz.plot_lag_band_summary(lag_summary, figs)
display(Image(filename=str(paths["png"])))

for measure in ("peak_excess", "positive_lag_excess", "negative_lag_excess"):
    display(Markdown(f"**{measure}**"))
    display(
        lag_summary.loc[lag_summary["measure"] == measure]
        .pivot_table(index="region_pair", columns="condition", values="mean").round(4)
    )

lag_contrasts = sc.compare_lag_measures(signal, signal_settings)
display(
    lag_contrasts.loc[
        lag_contrasts["measure"].isin(
            ["peak_excess", "positive_lag_excess", "negative_lag_excess"]
        ),
        ["measure", "region_pair", "condition_a", "condition_b", "n_pairs",
         "mean_difference", "effect_size_rank_biserial", "significant"],
    ].round(4)
)
'''

COMBINED = '''
joined = sc.join_with_noise_correlation(signal, signal_settings)
correlations = sc.correlate_signal_with_noise(joined)
print(f"pairs with both measurements: {len(joined):,}")
display(correlations.round(4))

from dal_monte_2022_analysis.ephys.plotting import fixation_signal_correlation as sigviz
sig_figs = sigviz.SignalCorrelationPlotSettings(output_dir=FIGURE_DIR)
fig, paths = sigviz.plot_signal_vs_noise(joined, correlations, sig_figs,
                                         stem="fig06_signal_vs_noise")
display(Image(filename=str(paths["png"])))
'''

TRIALS = '''
strata = sc.stratify_by_trial_ratio(signal, signal_settings, metric="peak_excess")
display(strata.round(4))

from dal_monte_2022_analysis.ephys.plotting import fixation_signal_correlation as sigviz
fig, paths = sigviz.plot_trial_count_confound(
    strata, sigviz.SignalCorrelationPlotSettings(output_dir=FIGURE_DIR),
    stem="fig07_trial_count_confound",
)
display(Image(filename=str(paths["png"])))
'''

_CELL_IDS = count(1)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"cell-{next(_CELL_IDS):02d}",
            "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": f"cell-{next(_CELL_IDS):02d}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    markdown(f"""
# {TITLE}

Every pair here is **two FDR-selective units recorded simultaneously in the same
region**. Two things are measured on those same pairs, and the point of putting
them together is that neither is interpretable alone.

| | noise correlation | signal correlation |
|---|---|---|
| computed on | per-fixation 1 ms spike trains | condition-averaged rate timelines |
| asks | do they fire together on the **same** fixation | do their **mean responses** share a shape |
| null | circular shift within fixation | unit of the same region, different session |

Averaging over fixations removes trial-by-trial covariation entirely. That is
the whole difference between the two rows of the schematic, and it is why a pair
can have either without the other.
"""),
    code(SETUP),
    code(SCHEMATIC),
    markdown("""
## 1. What is in the analysis
"""),
    code(LOAD),
    markdown("""
# Noise correlation

## 2. It sits above the null

Interactive-face fixations, per region: the observed cross-correlation against
the circular-shift null. The gap is the coordination.
"""),
    code(NOISE_ABOVE),
    markdown("""
## 3. But the fixation types do not differ

The same quantity, null-corrected, for all three conditions. Trial-count matched,
because interactive-face fixations outnumber the others roughly six to one.

Read the **effect sizes**, not the asterisks. With thousands of pairs per region
almost any difference reaches significance; rank-biserial is the number that
says whether it is a difference worth reporting.
"""),
    code(NOISE_EXCESS),
    markdown("""
# Signal correlation

## 4. Null-corrected correlation across lags

The cross-session null is subtracted, so zero means "resembles a same-region unit
from another session no more than chance".
"""),
    code(SIGNAL_TRACES),
    markdown("""
## 5. Summarised away from zero lag

Zero lag is one bin of fifty and a poor summary: two units whose responses share
a shape but differ in latency correlate strongly at a non-zero lag and weakly at
zero. Three measures instead —

- **peak**: the maximum over ±100 ms, i.e. similarity at best alignment
- **mean +20 to +200 ms** and **mean −20 to −200 ms**: whether that alignment is symmetric

Two caveats on reading these. The peak is a maximum over many noisy lags and is
biased upward in level — but identically for every condition, so comparisons
hold even though the absolute value does not. And within a region the ordering
of the two units in a pair is arbitrary, so the **sign** of a lead/lag difference
carries no meaning; the two bands are expected to be similar and it is a
departure that would be notable.
"""),
    code(SIGNAL_BANDS),
    markdown("""
# Putting them together

## 6. Do the two measures track each other?

Pairs that share a response profile need not covary trial to trial, and vice
versa. Matching each pair to both of its measurements answers it directly.
"""),
    code(COMBINED),
    markdown("""
## 7. The trial-count caveat, stated rather than used to dismiss

Interactive-face fixations outnumber the others about six to one, so
interactive-face mean timelines are estimated more precisely and correlate
better with anything. This inflates signal correlation for interactive face
specifically, and the cross-session null does not absorb it.

The noise-correlation results above are already trial-count matched, so they do
not carry this. The signal-correlation results are **not** matched — no matched
average exists — so the stratification below is the available control: shared
tuning predicts a difference flat across strata of the trial-count ratio,
estimation noise predicts one that grows with it.

This bounds how much of section 5 to believe. It does not make the analysis
worthless: the level of signal correlation over the cross-session null, the
region differences, and the signal/noise relationship in section 6 are all
comparisons that the imbalance does not obviously drive.
"""),
    code(TRIALS),
    markdown("""
## What the two analyses say together

Fill in against the numbers, but the shape of the answer:

1. **Noise correlation is clearly above the circular-shift null in every
   region** — simultaneously recorded selective pairs do co-fire beyond what
   their individual rate profiles predict.
2. **Fixation type does not modulate it.** Effect sizes across every region and
   contrast are a few hundredths, and the significance is sample size.
3. **Signal correlation is above its own null**, and unlike noise correlation it
   does vary between regions and shows larger condition effect sizes — most of
   it in OFC, which is also where the trial-count confound bites hardest.
4. **The two are positively related within region.** Pairs whose mean responses
   resemble each other also tend to co-fire trial to trial.

The honest headline is that pairwise coupling in this dataset is a property of
the **pair and the region**, not of what the animal was looking at.
"""),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / FILENAME
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
