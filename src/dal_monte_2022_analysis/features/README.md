# features

Feature construction modules that transform processed signals into analysis-ready products.

Modules:
- `gaze_event_detection.py`
  Detects fixations/saccades and annotates ROI labels.
- `fixation_binary_vectors.py`
  Converts fixation intervals into timeline-aligned binary vectors.
- `fixation_density.py`
  Applies Gaussian smoothing to build fixation density vectors.
- `joint_fixation_density.py`
  Combines m1 and m2 face-density traces into a joint density signal.
- `interactive_periods.py`
  Segments joint density into interactive and non-interactive periods.

All modules use `*_Settings` dataclasses and `run_*` entry functions to support script-level orchestration.
