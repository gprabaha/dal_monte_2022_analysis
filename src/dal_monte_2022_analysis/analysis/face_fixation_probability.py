"""Backwards-compatible wrapper for fixation probability analysis."""

from __future__ import annotations

from dal_monte_2022_analysis.analysis.fixation_probability import (
    FixationProbabilitySettings,
    run_fixation_probability_analysis,
    run_interactive_fixation_probability_analysis,
)


FaceFixationProbabilitySettings = FixationProbabilitySettings
run_face_fixation_probability_analysis = run_fixation_probability_analysis
run_interactive_face_fixation_probability_analysis = (
    run_interactive_fixation_probability_analysis
)

__all__ = [
    "FixationProbabilitySettings",
    "FaceFixationProbabilitySettings",
    "run_fixation_probability_analysis",
    "run_face_fixation_probability_analysis",
    "run_interactive_fixation_probability_analysis",
    "run_interactive_face_fixation_probability_analysis",
]
