# Combined Fixation Mean PSTH Data Dictionary

This document describes the two combined fixation mean PSTH files.

## Files

### 1) Combined dataframe pickle

Filename:

`fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl`

Contents:

- A single `pandas.DataFrame`
- Current export size: `7206` rows x `13` columns
- One row per averaged unit-condition entry
- Contains both split and unsplit condition averages in the same table

### 2) Timeline pickle

Filename:

`fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl`

Contents:

- A one-dimensional `numpy.ndarray`
- Relative time-bin centers in seconds
- Current export length: `100`
- Bin centers run from approximately `-0.495 s` to `+0.495 s`
- Mean step size is `0.01 s` (`10 ms`)

## Averaging Details

These combined files were built from the per-date average PSTH files:

- exported time window: extracted `-0.5 s` to `+0.5 s`
- source values: firing rate in Hz
- smoothing before averaging: enabled
- smoothing sigma: `20 ms`

## Split Versus Unsplit Semantics

- `average_partition == "split"`
  - both `face` and `object` rows are split by `interactive_state`
  - observed row groups are:
    - `face` + `interactive`
    - `face` + `non_interactive`
    - `object` + `interactive`
    - `object` + `non_interactive`

- `average_partition == "unsplit"`
  - neither `face` nor `object` is split by `interactive_state`
  - observed row groups are:
    - `face` + `interactive_state=None`
    - `object` + `interactive_state=None`

For the current export, the row counts are:

| `average_partition` | `fixation_category` | `interactive_state` | Row count |
| --- | --- | --- | ---: |
| `split` | `face` | `interactive` | 1201 |
| `split` | `face` | `non_interactive` | 1201 |
| `split` | `object` | `interactive` | 1201 |
| `split` | `object` | `non_interactive` | 1201 |
| `unsplit` | `face` | `None` | 1201 |
| `unsplit` | `object` | `None` | 1201 |

The split export contains split face and
split object activity, and the unsplit export contains unsplit face and unsplit
object activity.


## Dataframe Columns

| Column | Type in loaded row | Description |
| --- | --- | --- |
| `date` | `str` | Recording date identifier taken from the source date folder, e.g. `01022019`. |
| `unit_uuid` | `str` | Unique unit identifier for the recorded neuron/unit. |
| `region` | `str` or null-like | Canonical brain region label for the unit. Observed values in this export are `accg`, `bla`, `dmpfc`, and `ofc`. |
| `spike_channel` | `str` or null-like | Spike channel identifier associated with the unit. |
| `recorded_agent` | `str` | Agent from which the neural unit was recorded. In the current export this column is constant at `m1`. |
| `fixation_category` | `str` | Broad fixation condition label. Values in the current export are `face` and `object`. |
| `interactive_state` | `str` or `None` | Interactive-state label. Values are `interactive`, `non_interactive`, or `None` for unsplit rows. |
| `is_interactive` | `bool` or `None` | Boolean companion to `interactive_state`. `True` for interactive rows, `False` for non-interactive rows, and `None` for unsplit rows. |
| `n_trials` | `int` | Number of fixation-aligned trials contributing to the averaged PSTH row. |
| `psth_mean` | `numpy.ndarray` | Length-100 vector of mean firing-rate values aligned to the saved timeline. Units are Hz. |
| `psth_sem` | `numpy.ndarray` | Length-100 vector of standard-error values aligned to the saved timeline. Units are Hz. |
| `source_fixation_monkeys` | `tuple[str, ...]` | Sorted tuple of contributing fixation-monkey labels observed across trials contributing to the row. Example values include `('Cronenberg', 'Lynch')` and `('Lynch',)`. |
| `average_partition` | `str` | Indicates whether the row came from the split or unsplit source average table. Values are `split` or `unsplit`. |

## Timeline Variable

The timeline file stores the relative time axis used by every `psth_mean` and
`psth_sem` vector in the dataframe.

| Variable | Type | Description |
| --- | --- | --- |
| timeline array | `numpy.ndarray` | Length-100 vector of relative time-bin centers in seconds. |

Alignment rule:

- `timeline[i]` aligns to `psth_mean[i]` and `psth_sem[i]` for every row.
- The saved values are bin centers, not bin edges.
- Because the source data use 10 ms bins, the exported `[-500 ms, +500 ms]`
  window corresponds to centers at approximately `-495 ms, -485 ms, ..., +495 ms`.

## Notes

- rows are averaged PSTHs, not single-trial PSTHs
- values are firing rates in Hz
- the exported window is centered on fixation onset
- both split and unsplit averages are present in the same dataframe
- `average_partition` should be used to separate split rows from unsplit rows
- the separate timeline file must be used to interpret the PSTH vector index

## Minimal Loading Example

```python
import pickle
import pandas as pd

df = pd.read_pickle(
    "fixations_psth_10ms_combined_window_neg500ms_to_pos500ms.pkl"
)
timeline_s_rel = pickle.load(
    open("fixations_psth_10ms_bin_centers_s_rel_window_neg500ms_to_pos500ms.pkl", "rb")
)

split_face = df[
    (df["average_partition"] == "split")
    & (df["fixation_category"] == "face")
    & (df["interactive_state"] == "interactive")
]

row = split_face.iloc[0]
mean_trace_hz = row["psth_mean"]
sem_trace_hz = row["psth_sem"]
```
