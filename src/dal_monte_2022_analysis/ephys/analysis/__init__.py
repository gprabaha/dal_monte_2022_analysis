"""Ephys analysis modules."""

from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation import (
    CROSS_ANALYSIS_KIND,
    FixationNeuralCrossCorrelationPlotAggregationSettings,
    FixationNeuralCrossCorrelationSettings,
    WITHIN_ANALYSIS_KIND,
    build_cross_region_fixation_neural_cross_correlation_plot_payload,
    build_fixation_neural_cross_correlation_plot_payload,
    build_within_region_fixation_neural_cross_correlation_plot_payload,
    run_cross_region_fixation_neural_cross_correlation,
    run_within_region_fixation_neural_cross_correlation,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_selectivity import (
    FixationPSTHSelectivitySettings,
    run_fixation_selectivity_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_preference_index import (
    FixationPSTHPreferenceIndexSettings,
    run_fixation_preference_index_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonSettings,
    run_fixation_three_way_region_comparison,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pca import (
    FixationPopulationPCASettings,
    run_fixation_population_pca_analysis,
)

__all__ = [
    "FixationPSTHSelectivitySettings",
    "run_fixation_selectivity_analysis",
    "FixationPSTHPreferenceIndexSettings",
    "run_fixation_preference_index_analysis",
    "FixationThreeWayRegionComparisonSettings",
    "run_fixation_three_way_region_comparison",
    "FixationPopulationPCASettings",
    "run_fixation_population_pca_analysis",
    "WITHIN_ANALYSIS_KIND",
    "CROSS_ANALYSIS_KIND",
    "FixationNeuralCrossCorrelationSettings",
    "FixationNeuralCrossCorrelationPlotAggregationSettings",
    "run_within_region_fixation_neural_cross_correlation",
    "run_cross_region_fixation_neural_cross_correlation",
    "build_fixation_neural_cross_correlation_plot_payload",
    "build_within_region_fixation_neural_cross_correlation_plot_payload",
    "build_cross_region_fixation_neural_cross_correlation_plot_payload",
]
