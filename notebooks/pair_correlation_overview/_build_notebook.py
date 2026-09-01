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
noise = noise.loc[noise["both_selective"]].copy()
noise["scope"] = np.where(noise["same_region"], "within_region", "cross_region")
noise_traces = pd.read_pickle(SUMMARY_DIR / "traces_by_region_selective.pkl")
print(f"noise: {len(noise):,} pair-conditions, both units FDR-selective   "
      f"(artifact dates removed: {dropped})")

# --- signal correlation: condition-averaged timelines, cross-session null ----
signal_settings = sc.SignalCorrelationSettings(cfg_path=CFG_PATH)
units, timeline = sc.load_condition_timelines(signal_settings)
signal, signal_lags = sc.build_pair_correlations(units, timeline, signal_settings)
signal_traces = {
    "lags_ms": signal_lags,
    "traces": sc.build_group_traces(signal, signal_settings),
}
print(f"signal: {len(signal):,} pairs from {len(units)} FDR-selective units")

joined = sc.join_with_noise_correlation(
    signal, signal_settings, signal_metric=sc.WINDOW_METRIC
)
correlations = sc.correlate_signal_with_noise(joined)

display(
    signal.groupby(["scope", "region_pair"], observed=True)
    .size().rename("signal_pairs").to_frame()
)
'''


def signal_traces_cell(scope: str) -> str:
    return f'''
fig, paths = viz.plot_excess_by_condition(
    signal_traces, figs, scope="{scope}", max_lag_ms=250.0,
    ylabel="Signal correlation\\n(observed − null)",
    title="Signal correlation, null-corrected",
    stem="fig02_signal_excess",
)
display(Image(filename=str(paths["png"])))
'''


def noise_cell(scope: str) -> str:
    return f'''
fig, paths = viz.plot_noise_above_null(noise_traces, figs, scope="{scope}")
display(Image(filename=str(paths["png"])))
'''


def bars_cell(scope: str) -> str:
    return f'''
summary = sc.summarize_lag_measures(
    signal, signal_settings, measures=(sc.WINDOW_METRIC,), scope="{scope}"
)
contrasts = sc.compare_lag_measures(
    signal, signal_settings, measures=(sc.WINDOW_METRIC,), scope="{scope}"
)
rho = correlations.loc[correlations["scope"] == "{scope}"]

fig, paths = viz.plot_summary_bars(summary, contrasts, rho, figs, scope="{scope}")
display(Image(filename=str(paths["png"])))

display(
    summary.pivot_table(index="region_pair", columns="condition", values="mean").round(4)
)
display(
    contrasts.loc[
        :, ["region_pair", "condition_a", "condition_b", "n_pairs", "mean_difference",
            "effect_size_rank_biserial", "p_value_corrected", "significant"]
    ].round(4)
)
display(rho.loc[:, ["region_pair", "condition", "n_pairs", "spearman_rho", "p_value"]].round(4))
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

Every pair here is **two FDR-selective units recorded simultaneously**. Two
things are measured on those same pairs.

| | signal correlation | noise correlation |
|---|---|---|
| computed on | condition-averaged rate timelines | per-fixation 1 ms spike trains |
| asks | do their **mean responses** share a shape | do they fire together on the **same** fixation |
| null | unit of the same region, different session | circular shift within fixation |
| units | Pearson coefficient, −1 to 1 | spike pairs per fixation, unnormalised |

Averaging over fixations removes trial-by-trial covariation entirely. That is
the only difference between the two paths in the schematic, and it is why a pair
can have either without the other.

Within-region pairs come first, then the three cross-region combinations the
recordings support — all of which involve BLA, since ACCg and OFC were never
recorded together and dmPFC × OFC comes from a handful of sessions.
"""),
    code(SETUP),
    code(SCHEMATIC),
    markdown("""
## What is in the analysis
"""),
    code(LOAD),
    markdown("""
# Within region

## 1. Signal correlation

The cross-session null is subtracted, so zero means "resembles a same-region
unit from another session no more than chance".
"""),
    code(signal_traces_cell("within_region")),
    markdown("""
## 2. Noise correlation

Rows are fixation conditions, columns are regions. The gap between the two
curves is the coordination — the observed curve alone is not, since an
unnormalised cross-correlation scales with the product of the two firing rates.

The three rows look alike, which is the result: fixation type does not modulate
noise correlation.
"""),
    code(noise_cell("within_region")),
    markdown("""
## 3. Summary

**Left**: null-corrected signal correlation, summarised as the mean over ±100 ms
— not at zero lag, which is one bin of fifty, and not at the peak, whose
per-pair maximum falls at a different lag for every pair and so averages to a
much larger number than the maximum of the average. The windowed mean takes no
maximum anywhere, so these bars and the traces in section 1 are the same
quantity. Brackets appear only where a contrast survives FDR.

**Right**: the Spearman correlation between each pair's signal and noise
correlation. A positive value means pairs whose mean responses resemble each
other also tend to co-fire trial to trial — which is not guaranteed, since the
two come from different operations on the same trains.
"""),
    code(bars_cell("within_region")),
    markdown("""
# Across regions

Only BLA × ACCg, BLA × dmPFC and BLA × OFC are populated enough to report.

## 4. Signal correlation
"""),
    code(signal_traces_cell("cross_region")),
    markdown("""
## 5. Noise correlation
"""),
    code(noise_cell("cross_region")),
    markdown("""
## 6. Summary
"""),
    code(bars_cell("cross_region")),
    markdown("""
## What the two analyses say together

1. **Signal correlation is above its cross-session null**, and it varies by
   region and by fixation type.
2. **Noise correlation is clearly above its circular-shift null**, so
   simultaneously recorded selective pairs do co-fire beyond what their
   individual rate profiles predict.
3. **Fixation type does not modulate noise correlation** — the three rows of the
   noise grid are alike in every region.
4. **The two measures are positively related within region.**

So the fixation-condition effect lives in **shared tuning**, not in
trial-by-trial coupling: what changes with interactive face is how much two
neurons' average responses resemble each other, not how tightly they co-fire on
any given fixation.

**One caveat to carry.** Interactive-face fixations outnumber the others roughly
six to one, so interactive-face mean timelines are estimated more precisely and
correlate better with anything. The noise comparisons are trial-count matched
and do not carry this; the signal comparisons cannot be, so their sizes are
upper bounds. `notebooks/signal_correlation/` stratifies by the trial-count
ratio to bound it directly.
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
