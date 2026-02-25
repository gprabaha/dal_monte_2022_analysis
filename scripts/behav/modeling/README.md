# scripts/behav/modeling

CLI entrypoints for latent-state model fitting.

Scripts:
- `build_face_fixation_hsmm.py`
  Fits a 2-state Poisson hidden semi-Markov model to joint face-fixation
  observations (`(m1, m2)` in `{(0,0), (1,0), (0,1), (1,1)}`), with grouping
  controls (`session`, `day`, `pair`, `global`).

Primary config:
- `configs/face_fixation_hsmm.yaml`
