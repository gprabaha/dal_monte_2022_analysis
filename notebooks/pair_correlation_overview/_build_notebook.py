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

SIGNAL_PEAK = '''
lag_summary = sc.summarize_lag_measures(
    signal, signal_settings, measures=(sc.WINDOW_METRIC,)
)
peak_contrasts = sc.compare_lag_measures(
    signal, signal_settings, measures=(sc.WINDOW_METRIC,)
)

fig, paths = viz.plot_peak_comparison(
    lag_summary, figs, contrasts=peak_contrasts,
    title="Signal correlation by region and fixation type",
)
display(Image(filename=str(paths["png"])))

display(
    lag_summary.pivot_table(index="region_pair", columns="condition", values="mean").round(4)
)
display(
    peak_contrasts.loc[
        :, ["region_pair", "condition_a", "condition_b", "n_pairs", "mean_difference",
            "effect_size_rank_biserial", "p_value_corrected", "significant"]
    ].round(4)
)
'''

COMBINED = '''
joined = sc.join_with_noise_correlation(signal, signal_settings)
joined = joined.loc[joined["scope"] == "within_region"]
correlations = sc.correlate_signal_with_noise(joined)
print(f"pairs with both measurements: {len(joined) // 3:,} per condition")

fig, paths = viz.plot_peak_signal_vs_noise(joined, correlations, figs)
display(Image(filename=str(paths["png"])))

display(
    correlations.loc[
        :, ["region_pair", "condition", "n_pairs", "spearman_rho", "p_value"]
    ].round(4)
)
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
region**. Two things are measured on those same pairs.

| | signal correlation | noise correlation |
|---|---|---|
| computed on | condition-averaged rate timelines | per-fixation 1 ms spike trains |
| asks | do their **mean responses** share a shape | do they fire together on the **same** fixation |
| null | unit of the same region, different session | circular shift within fixation |

Averaging over fixations removes trial-by-trial covariation entirely. That is
the only difference between the two paths in the schematic, and it is why a pair
can have either without the other.

Signal correlation comes first because it carries the clearer result.
"""),
    code(SETUP),
    code(SCHEMATIC),
    markdown("""
## 1. What is in the analysis
"""),
    code(LOAD),
    markdown("""
# Signal correlation

## 2. Null-corrected correlation across lags

The cross-session null is subtracted, so zero means "resembles a same-region
unit from another session no more than chance".
"""),
    code(SIGNAL_TRACES),
    markdown("""
## 3. Correlation by region and fixation type

Summarised as the **mean over ±100 ms**, not at zero lag and not at the peak.

- *Not zero lag*: it is one bin of fifty, and two units whose responses share a
  shape but differ in latency correlate strongly off zero and weakly at it.
- *Not the peak*: each pair's maximum falls at a different lag, so averaging
  per-pair maxima gives a far larger number than the maximum of the average —
  a bar chart of peaks reads around 0.3 while the trace above it peaks near
  0.1. The windowed mean has no maximum in it, so the bars and the traces are
  the same quantity: the mean over pairs of this **equals** the mean of the
  group trace over the same window.

Brackets carry the rank-biserial effect size, starred where the contrast
survives FDR correction. With hundreds to thousands of pairs per region almost
any difference clears an alpha, so the star says "survived correction" and the
number says whether it matters.
"""),
    code(SIGNAL_PEAK),
    markdown("""
# Noise correlation

## 4. It sits above the null, and fixation type does not change it

Interactive face first, against the circular-shift null. The gap between the two
curves is the coordination.

The y-axis is **coincidences per fixation**, not a correlation coefficient: at
each 1 ms lag it counts spike pairs separated by that lag. Chance is roughly
`rate₁ × rate₂ × bin width`, about 0.05 for two 7 Hz units, which is why the
values sit where they do — and why they are not comparable in magnitude with the
signal correlation above.
"""),
    code(NOISE_ABOVE),
    markdown("""
The same quantity null-corrected, for all three conditions, trial-count matched.
Read the **effect sizes**: with thousands of pairs per region almost any
difference reaches significance.
"""),
    code(NOISE_EXCESS),
    markdown("""
# Putting them together

## 5. Peak signal against peak noise, per region

One panel per region, since the two quantities differ by an order of magnitude
between regions and a pooled scatter would show mostly that. Points are coloured
by fixation condition and the Spearman correlation per condition is printed on
the panel.
"""),
    code(COMBINED),
    markdown("""
## What the two analyses say together

1. **Signal correlation is above its cross-session null**, and it varies by
   region and by fixation type — interactive face is highest where the effect
   is clearest.
2. **Noise correlation is clearly above its circular-shift null in every
   region**, so simultaneously recorded selective pairs do co-fire beyond what
   their individual rate profiles predict.
3. **Fixation type does not modulate noise correlation.** Effect sizes across
   every region and contrast are a few hundredths.
4. **The two are positively related within region** — pairs whose mean responses
   resemble each other also tend to co-fire trial to trial.

So the fixation-condition effect lives in **shared tuning**, not in
trial-by-trial coupling: what changes with interactive face is how much two
neurons' average responses resemble each other, not how tightly they co-fire on
any given fixation.

One caveat to carry: interactive-face fixations outnumber the others roughly six
to one. The noise-correlation comparisons are trial-count matched and do not
carry that; the signal-correlation comparisons cannot be, so their absolute
sizes are upper bounds even though the region and condition ordering is not
obviously driven by it.
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
