# Noise and signal correlation in simultaneously recorded selective pairs

Written as a **thesis chapter**: introduction, methods, results and discussion,
with the figures inline. Figures are also written as editable PDFs for stitching
into a paper figure.

The combined analysis. Every pair is **two FDR-selective units recorded
simultaneously in the same region**, and two things are measured on those same
pairs.

| | noise correlation | signal correlation |
|---|---|---|
| computed on | per-fixation 1 ms spike trains | condition-averaged rate timelines |
| asks | do they fire together on the **same** fixation | do their **mean responses** share a shape |
| null | circular shift within fixation | unit of the same region, different session |
| trial-count matched | yes | no — see caveat |

Averaging over fixations removes trial-by-trial covariation entirely. That is
the only difference between the two, and it is why a pair can have either
without the other.

## Structure

Signal correlation comes first because it carries the clearer result.

1. Method schematic — one set of trials, two orders of operation
2. **Signal**: null-corrected correlation across lags
3. **Signal**: peak height by region and fixation type, with significance
4. **Noise**: observed against null, every region and condition
5. **Noise**: null-subtracted — the fixation types do not differ
6. Spearman correlation between the two measures, per region and condition

## How the signal-correlation bars are computed

Summarising at a peak is the right instinct and has two wrong implementations.

*A per-pair maximum* inflates the level: every pair peaks at a different lag, so
the mean of the maxima far exceeds the maximum of the mean. Bars built that way
read ≈0.30 beside traces peaking at ≈0.10 — same data, two incompatible numbers.

*A fixed window* (the mean over ±100 ms) is unbiased and matches the trace
exactly, but it is not the peak: it averages the peak with its shoulders and
sits below the maximum a reader can see.

What is used instead: the peak lag is found **once per group from the group-mean
trace**, searched over the full ±250 ms, and every pair is read at that single
lag. No maximum is taken per pair, so the bar equals the visible peak of the
trace, and because each pair contributes a value at a fixed lag the per-pair
spread and the paired tests stay valid.

## Why the two y-axes are not comparable

Signal correlation is a Pearson coefficient, bounded in [−1, 1]. Noise
correlation is **coincidences per fixation**: at each 1 ms lag, the number of
spike pairs separated by that lag. Chance is roughly `rate₁ × rate₂ × bin
width` — about 0.05 for two 7 Hz units — which is why those values sit where
they do. They are different units and only their *ranks* are compared, in the
final scatter.

## The trial-count caveat

Interactive-face fixations outnumber the others about six to one, so
interactive-face mean timelines are estimated more precisely and correlate
better with anything. The noise-correlation comparisons are trial-count matched
and do not carry this; the signal-correlation comparisons cannot be, so their
absolute sizes are upper bounds. The region and condition ordering is not
obviously driven by it. `notebooks/signal_correlation/` has the stratification
that bounds it directly.

## Reading the statistics

With thousands of pairs per region almost any difference reaches significance.
**Read the rank-biserial effect sizes, not the asterisks.**
