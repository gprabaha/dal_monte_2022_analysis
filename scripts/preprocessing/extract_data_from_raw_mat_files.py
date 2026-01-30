"""Extract raw .mat files into per-session pickles per modality using configs/dataset.yaml."""

from dal_monte_2022_analysis.io.build_dataset import build_agent_dataset
from dal_monte_2022_analysis.io.data_extraction import (
    extract_position,
    extract_neural_timeline,
    extract_pupil,
    extract_roi_rects,
)

# Toggle multiprocessing on/off for debugging versus full-speed runs.
USE_PARALLEL = True

# Position samples (agent-specific).
build_agent_dataset(
    cfg_path="configs/dataset.yaml",
    modality="gaze_position",
    extractor_fn=extract_position,
    agent_specific=True,
    use_parallel=USE_PARALLEL,
)

# Timeline (shared across agents).
build_agent_dataset(
    cfg_path="configs/dataset.yaml",
    modality="neural_timeline",
    extractor_fn=extract_neural_timeline,
    agent_specific=False,
    use_parallel=USE_PARALLEL,
)

# Pupil size samples (agent-specific).
build_agent_dataset(
    cfg_path="configs/dataset.yaml",
    modality="pupil_size",
    extractor_fn=extract_pupil,
    agent_specific=True,
    use_parallel=USE_PARALLEL,
)

# ROI rectangles (agent-specific).
build_agent_dataset(
    cfg_path="configs/dataset.yaml",
    modality="roi_vertices",
    extractor_fn=extract_roi_rects,
    agent_specific=True,
    use_parallel=USE_PARALLEL,
)
