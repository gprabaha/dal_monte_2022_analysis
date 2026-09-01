# Spatial decay of pairwise spike coordination

Pairwise spike coordination falls off steeply with the distance between the two
electrodes. This notebook measures the fall-off, fits a length constant, and
asks whether it depends on what the animal was fixating.

## Contents

- `pair_spatial_decay.ipynb` — the analysis notebook.
- `_build_notebook.py` — authors it from source strings. Edit this, never the
  `.ipynb`, then re-run it.

```bash
conda run -n gaze_processing python notebooks/pair_spatial_decay/_build_notebook.py
```

It reads the per-session outputs built by
[`../pair_spike_coordination/`](../pair_spike_coordination/) and computes
nothing from raw data, so that pipeline must have been run first.

## Why this analysis exists

Within-region pairs are recorded on the **same electrode array**; cross-region
pairs are not. So "within-region pairs are more coordinated than cross-region
pairs" is confounded before it starts — a shared reference, a shared amplifier
or any common noise on an array produces that pattern with no biology involved.

Electrode separation breaks the tie, because the two explanations make opposite
predictions about the same measurement:

| hypothesis | prediction |
|---|---|
| shared reference / common noise | coordination **flat** with separation |
| local circuitry | coordination **decays** with separation |

No new data is needed; the discriminating test is already in the pairs we have.

## The separation measure, and what it is not

Channels are named `SPKnn` and numbered in contiguous blocks per region, so
`|n1 − n2|` is the separation. This is an **uncalibrated proxy** for physical
distance: it assumes channel numbering runs in spatial order. A monotone decay
is itself evidence that the numbering tracks something spatial — an arbitrary
permutation of channel labels would destroy it — but that is an argument, not a
calibration. **Check it against the array geometry before quoting length
constants in millimetres.**

Same-channel pairs are excluded and reported separately: they carry a *negative*
zero-lag excess, the spike sorter's inability to assign two spikes to different
units in the same millisecond. Its appearing with the right sign is a check that
the pipeline measures what it claims.

## What is measured

Per-fixation cross-correlations on unsmoothed 1 ms trains over ±500 ms, against
a circular-shift null, summarised per pair by a **sharp peak** (±2 ms) and a
**broad shoulder** (20–100 ms), each baselined on that pair's own 200–250 ms
flank. Curves are fitted as `A·exp(−d/λ) + c`, with `c` fitted rather than
assumed zero so that any residual same-array offset stays separable from the
amplitude.

## Caveats

1. Channel separation is an uncalibrated proxy for distance.
2. BLA has no pairs beyond 15 channels apart, so its long-range offset is the
   least constrained of the four fits.
3. The far within-region level sits somewhat above the cross-region level; some
   same-array offset may remain after the decay is accounted for.
4. ACCg has both the shallowest decay and the narrowest correlation peak —
   "spatially flat and millisecond wide" is also what a residual common-noise
   contribution would look like, so it deserves scrutiny.

## Outputs

Figures are written to
`analysis_output_root/ephys/psth/fixation_pair_spike_coordination/spatial_decay/`
as editable PDF plus high-resolution PNG.
