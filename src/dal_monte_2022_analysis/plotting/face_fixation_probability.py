"""Backwards-compatible wrapper for fixation probability plotting."""

from __future__ import annotations

from dal_monte_2022_analysis.plotting.fixation_probability import (
    FixationProbabilityPlotSettings,
    InteractiveFixationProbabilityPlotSettings,
    plot_fixation_probability_violin,
    plot_interactive_fixation_probability_violin,
)


FaceFixationProbabilityPlotSettings = FixationProbabilityPlotSettings
InteractiveFaceFixationProbabilityPlotSettings = (
    InteractiveFixationProbabilityPlotSettings
)
plot_face_fixation_probability_violin = plot_fixation_probability_violin
plot_interactive_face_fixation_probability_violin = (
    plot_interactive_fixation_probability_violin
)

__all__ = [
    "FixationProbabilityPlotSettings",
    "InteractiveFixationProbabilityPlotSettings",
    "plot_fixation_probability_violin",
    "plot_interactive_fixation_probability_violin",
    "FaceFixationProbabilityPlotSettings",
    "InteractiveFaceFixationProbabilityPlotSettings",
    "plot_face_fixation_probability_violin",
    "plot_interactive_face_fixation_probability_violin",
]
