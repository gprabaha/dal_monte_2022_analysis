# modeling

Latent-state modeling modules built on top of processed feature vectors.

Modules:
- `face_fixation_hsmm.py`
  Fits a 2-state hidden semi-Markov model (Poisson dwell-time distribution)
  to joint m1/m2 face-fixation observations with configurable grouping
  (`session`, `day`, `pair`, `global`).

Primary entrypoint:
- `run_face_fixation_hsmm_analysis(settings)`

Observation encoding:
- `0`: `(m1=0, m2=0)`
- `1`: `(m1=1, m2=0)`
- `2`: `(m1=0, m2=1)`
- `3`: `(m1=1, m2=1)`
