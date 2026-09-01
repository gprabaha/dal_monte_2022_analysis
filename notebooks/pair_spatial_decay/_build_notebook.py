"""Author the spatial-decay notebook from source strings.

Per ``AGENTS.md`` the notebook is thin: every function it calls lives in
``src/dal_monte_2022_analysis``.

    conda run -n gaze_processing python notebooks/pair_spatial_decay/_build_notebook.py
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

FILENAME = "pair_spatial_decay.ipynb"
TITLE = "Spatial decay of pairwise spike coordination"


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
from dal_monte_2022_analysis.ephys.analysis import fixation_pair_spatial_decay as sd
from dal_monte_2022_analysis.ephys.plotting import fixation_pair_spatial_decay as viz

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

cfg = load_config(str(repo_root / "configs" / "dataset.yaml"))
settings = sd.SpatialDecaySettings(cfg_path=str(repo_root / "configs" / "dataset.yaml"))
FIGURE_DIR = build_analysis_output_dir(cfg, sd.DEFAULT_OUTPUT_SUBDIR) / "spatial_decay"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figs = viz.SpatialDecayPlotSettings(output_dir=FIGURE_DIR)
print("figures ->", FIGURE_DIR)
'''

CONFOUND_FIG = '''
fig, paths = viz.plot_confound_schematic(figs)
display(Image(filename=str(paths["png"])))
'''

METHOD_FIG = '''
fig, paths = viz.plot_method_schematic(figs)
display(Image(filename=str(paths["png"])))
'''

LOAD = '''
pairs, dropped = sd.load_pairs_with_separation(settings)
print(f"pairs: {len(pairs):,}   artifact dates removed: {dropped}")
print(f"within region: {(pairs['scope'] == 'within_region').sum():,}   "
      f"across regions: {(pairs['scope'] == 'cross_region').sum():,}   "
      f"same channel: {pairs['same_channel'].sum():,}")

display(sd.build_separation_inventory(pairs))
'''

DECAY = '''
decay = sd.build_decay_table(pairs, settings)
fits = sd.fit_decay_by_group(decay)
references = sd.build_reference_levels(pairs)

display(decay.round(5))
fig, paths = viz.plot_decay_curves(decay, fits, figs, references=references)
display(Image(filename=str(paths["png"])))
'''

FLATNESS = '''
display(sd.test_decay_flatness(decay).round(5))
display(references.round(5))
'''

PARAMETERS = '''
display(fits.round(4))
fig, paths = viz.plot_fit_parameters(fits, figs)
display(Image(filename=str(paths["png"])))
'''

BY_CONDITION = '''
decay_by_condition = sd.build_decay_table(pairs, settings, by_condition=True)
fits_by_condition = sd.fit_decay_by_group(
    decay_by_condition, group_columns=("region", "condition")
)
display(fits_by_condition.round(4))

fig, paths = viz.plot_decay_by_condition(decay_by_condition, fits_by_condition, figs)
display(Image(filename=str(paths["png"])))

fig, paths = viz.plot_fit_parameters(fits_by_condition, figs, by_condition=True)
display(Image(filename=str(paths["png"])))
'''

CONDITION_TREND = '''
trend = sd.test_condition_by_separation(pairs)
display(trend.round(5))

fig, paths = viz.plot_condition_by_separation(trend, figs)
display(Image(filename=str(paths["png"])))

n_sig = int(trend["significant"].sum())
print(f"{n_sig} of {len(trend)} region x separation cells reach significance, "
      f"max |effect size| = {trend['effect_size_rank_biserial'].abs().max():.3f}")
'''

SHOULDER = '''
shoulder = sd.build_decay_table(pairs, settings, metric=sd.SHOULDER_METRIC)
shoulder_fits = sd.fit_decay_by_group(shoulder)
display(shoulder_fits.round(4))

fig, paths = viz.plot_decay_curves(
    shoulder, shoulder_fits, figs,
    references=sd.build_reference_levels(pairs, metric=sd.SHOULDER_METRIC),
    metric=sd.SHOULDER_METRIC, stem="fig07_decay_curves_shoulder",
)
display(Image(filename=str(paths["png"])))
'''

SELECTIVE = '''
selective_settings = sd.SpatialDecaySettings(
    cfg_path=settings.cfg_path, selective_only=True
)
selective_pairs, _ = sd.load_pairs_with_separation(selective_settings)
print(f"pairs with both units FDR-selective: {len(selective_pairs):,}")

selective_decay = sd.build_decay_table(selective_pairs, selective_settings)
selective_fits = sd.fit_decay_by_group(selective_decay)
display(selective_fits.round(4))

comparison = fits.merge(selective_fits, on="region", suffixes=("_all", "_selective"))
display(
    comparison.loc[
        :, ["region", "amplitude_all", "amplitude_selective",
            "length_constant_all", "length_constant_selective"]
    ].round(4)
)
'''


_CELL_IDS = count(1)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-{next(_CELL_IDS):02d}",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"cell-{next(_CELL_IDS):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(f"""
# {TITLE}

Pairwise spike coordination falls off steeply with the distance between the two
electrodes. This notebook measures that fall-off, fits a length constant to it,
and asks whether it depends on what the animal was fixating.

## Why this analysis exists

Within-region pairs are recorded on the **same electrode array**; cross-region
pairs are not. So the observation that within-region pairs are more coordinated
than cross-region pairs is confounded before it starts — a shared reference, a
shared amplifier, or any common noise on an array produces exactly that pattern
with no biology involved. On its own the comparison cannot be interpreted.

Electrode separation breaks the tie, because the two explanations make
**opposite predictions about the same measurement**:

- a shared reference contaminates every pair on an array equally → **flat** with separation
- local circuitry → **decays** with separation

No new data is needed; the discriminating test is already in the pairs we have.
"""),
    code(SETUP),
    code(CONFOUND_FIG),
    markdown("""
## Methods

Every cross-correlation is computed on one fixation's two **unsmoothed 1 ms**
spike trains over the **±500 ms** window around fixation onset, then averaged
across fixations. The null is a **circular shift**: unit B's train is rotated by
at least 50 ms within its own fixation, so each fixation keeps its own spike
counts and slow envelope and only fine temporal alignment is destroyed. Two
cells whose excitability merely rises and falls together across fixations
therefore produce no excess.

The null-corrected correlation is summarised by two numbers per pair, because
they can behave differently: a **sharp peak** (±2 ms) and a **broad shoulder**
(20–100 ms), each after subtracting that pair's own 200–250 ms baseline.
"""),
    code(METHOD_FIG),
    markdown("""
### The separation measure, and what it is not

Channels are named `SPKnn` and numbered in contiguous blocks per region, so
`|n1 − n2|` is the separation. This is a **proxy** for physical distance, not a
calibrated one: it assumes channel numbering runs in spatial order. A monotone
decay is itself evidence that the numbering tracks something spatial — an
arbitrary permutation of channel labels would destroy it — but that is an
argument, not a calibration. **Check it against the array geometry before
quoting length constants in millimetres.**

Pairs on the *same* channel are excluded and reported separately. They carry a
*negative* zero-lag excess, because a spike sorter cannot assign two spikes to
different units in the same millisecond on one channel. That shadowing is a
property of the sorter, not the tissue — and its appearing with the right sign
is a check that the pipeline measures what it claims.
"""),
    code(LOAD),
    markdown("""
## 1. The decay

Each point is the mean null-corrected sharp peak for pairs in that separation
bin, with a bootstrap interval. The curve is a fitted exponential
`A·exp(−d/λ) + c`. The offset `c` is fitted rather than assumed zero: it is
where coordination settles at long range, and any shared-reference contribution
would live there rather than in the amplitude, which keeps the two separable.

The y-axis is logarithmic — the range spans two orders of magnitude across
regions, and a linear axis renders ACCg as a flat line at the bottom.
"""),
    code(DECAY),
    markdown("""
### Is it flat, as a shared reference would predict?

A Spearman correlation between separation and the per-bin mean. Flatness is the
artifact hypothesis; a strong negative rank correlation rejects it.

The reference levels below are what the decay should be read against: the
same-channel shadowing floor, the far within-region bin, and the cross-region
level. The far within-region pairs land close to — but still above — cross
region, which is the residual that a same-array offset would explain.
"""),
    code(FLATNESS),
    markdown("""
## 2. How strongly, and how far

Splitting the fit into amplitude and length constant makes the result readable
in one line. Watch which of the two carries the region differences.
"""),
    code(PARAMETERS),
    markdown("""
## 3. Is the decay fixation-specific?

The decay could depend on fixation condition in two distinct ways, and they
need separating:

1. the **level** differs by condition at every distance → curves shift vertically → *amplitude* changes
2. the **reach** differs → curves change shape → *length constant* changes

Fitting each condition separately tests both at once. Compare the bootstrap
intervals: overlapping intervals mean the parameter is not distinguishable
between conditions.
"""),
    code(BY_CONDITION),
    markdown("""
### Does the condition difference track separation?

A third possibility: the conditions might differ only at short range, where
coordination is strong, and converge at long range. This computes the
within-pair condition difference *inside* each separation bin, so such a trend
would show up across bins.

Trial-count matched, because interactive-face fixations outnumber the others
roughly six to one and an unmatched comparison mostly reflects that. Effect
sizes rather than means: at these sample sizes a mean difference of a few
ten-thousandths reaches significance and says nothing.
"""),
    code(CONDITION_TREND),
    markdown("""
## 4. Does the broad shoulder decay too?

The sharp peak and the broad shoulder could have different spatial extents —
millisecond synchrony and slower co-fluctuation need not travel the same
distance. If the shoulder decays with a similar length constant, the two are
plausibly the same underlying local structure seen at two timescales.
"""),
    code(SHOULDER),
    markdown("""
## 5. Does the selective subset behave differently?

Restricting to pairs where **both** units are FDR-selective for at least one
fixation-condition contrast. Selecting units by a condition contrast and then
asking about condition effects would be circular, but that is not what is asked
here — the question is whether the *spatial architecture* looks different in the
selective subpopulation, which selection on condition does not predetermine.
"""),
    code(SELECTIVE),
    markdown("""
## What to take from this

Fill in against the numbers above:

- **The decay is real, not an array offset.** Coordination falls monotonically
  with electrode separation in every region, which a shared reference cannot
  produce.
- **Regions differ in amplitude, not in reach.** Compare the two panels of the
  fit-parameter figure.
- **The architecture does not depend on the fixation condition.** Compare the
  bootstrap intervals across conditions in section 3.

### Caveats worth carrying into any write-up

1. Channel separation is an uncalibrated proxy for distance.
2. BLA has no pairs beyond 15 channels apart, so its long-range offset is the
   least constrained of the four fits.
3. The far within-region level sits somewhat above the cross-region level; some
   same-array offset may remain even after the decay is accounted for.
4. ACCg is the shallowest decay and also the region with the narrowest
   correlation peak — worth scrutiny, since "spatially flat and millisecond
   wide" is also what a residual common-noise contribution would look like.
"""),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = Path(__file__).resolve().parent / FILENAME
    out_path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out_path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
