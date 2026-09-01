# Signal correlation between condition-averaged rate timelines

Do two units' **mean responses to a fixation have the same shape**, and does
that depend on what the animal was looking at?

## How this differs from `../pair_spike_coordination/`

| | spike coordination | this notebook |
|---|---|---|
| input | per-fixation 1 ms spike trains | condition-averaged 10 ms rate timelines |
| measures | do they fire together on the *same* fixation | do their mean responses have the same *shape* |
| quantity | **noise** correlation | **signal** correlation |
| null | circular shift within fixation | cross-session unit pairing |

Averaging over fixations removes trial-by-trial covariation entirely, so nothing
here is noise correlation. The two are independent: a pair can share a response
profile and be independent trial to trial, or the reverse. Section 5 asks
whether they are related in this data.

## Why the null has to be another unit

Every unit is fixation-locked, so every mean timeline has structure around
fixation onset and any two of them correlate before shared tuning is involved. A
null that only scrambles time would confirm that trivial structure rather than
test the interesting claim.

The null is a **cross-session pairing**: unit A against a unit of the same region
recorded on a *different* session — fixation-locked, real, same pipeline, but
sharing no session, array or behaviour.

## The confound that decides this analysis

A mean timeline estimated from 168 fixations is noisier than one from 1169, and
a correlation between two noisy estimates is attenuated towards zero.
Interactive-face fixations outnumber the others ~6:1, so interactive-face
timelines are the cleanest and correlate best with anything.

The cross-session null does **not** absorb this — the null partner shares no
tuning, so its correlation sits near zero whatever the trial count.

Spearman's correction for attenuation is the textbook remedy and is **not usable
here**: it needs each timeline's reliability, and these means are smoothed before
averaging while the SEMs are not correspondingly reduced, so the estimate comes
out negative for most units. `estimate_timeline_reliability` computes it anyway
as a diagnostic that reports the problem rather than hiding it.

**Stratification is used instead.** The interactive-to-object trial-count ratio
ranges from ~1 to ~16 across pairs, so the comparison is repeated inside strata:
shared tuning predicts a flat difference, estimation noise predicts one that
grows with the ratio.

## Running it

```bash
conda run -n gaze_processing python notebooks/signal_correlation/_build_notebook.py
```

Reads the combined ±500 ms 10 ms-binned condition averages — the same export the
population PCA and mRNN use — plus the selective-unit calls and, for section 5,
the per-session outputs from `../pair_spike_coordination/`.

## Caveats

1. Trial-count imbalance is the dominant methodological problem; stratification
   controls it but does not remove it. A properly matched comparison needs the
   per-trial data re-averaged at matched counts.
2. Cross-region pairs are far fewer and noisier; treat those panels as weak.
3. The reliability estimate is negative for most units because the timelines are
   smoothed. Diagnostic only.
