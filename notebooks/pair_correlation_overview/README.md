# Noise and signal correlation in simultaneously recorded selective pairs

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

1. Method schematic — both computations side by side
2. **Noise**: interactive face sits above the circular-shift null, per region
3. **Noise**: null-corrected, all three conditions — they do not differ
4. **Signal**: null-corrected correlation across lags
5. **Signal**: summarised away from zero lag — peak, and the ±20–200 ms bands
6. Do the two measures track each other?
7. The trial-count caveat

## Why not zero lag for signal correlation

Zero lag is one bin of fifty and a poor summary: two units whose responses share
a shape but differ in latency correlate strongly at a non-zero lag and weakly at
zero. Reported instead: the **peak** over ±100 ms (similarity at best alignment)
and the **mean over +20 to +200 ms and −20 to −200 ms** (whether that alignment
is symmetric).

Two caveats on those. The peak is a maximum over many noisy lags and is inflated
in level — identically for every condition, so comparisons hold even though the
absolute value does not. And within a region the ordering of the two units is
arbitrary, so the **sign** of a lead/lag difference carries no meaning; the two
bands are expected to be similar and a departure would be the notable thing.

## The trial-count caveat

Interactive-face fixations outnumber the others about six to one, so
interactive-face mean timelines are estimated more precisely and correlate
better with anything. The cross-session null does not absorb this.

The **noise** results are trial-count matched and do not carry it. The
**signal** results are not — no matched average exists — so section 7
stratifies by the trial-count ratio instead: shared tuning predicts a difference
flat across strata, estimation noise predicts one that grows with it.

This bounds how much of section 5 to believe. It does not make the analysis
worthless: the level of signal correlation over its null, the region
differences, and the signal/noise relationship are comparisons the imbalance
does not obviously drive.

## Reading the statistics

With thousands of pairs per region almost any difference reaches significance.
**Read the rank-biserial effect sizes, not the asterisks.**
