"""Ephys plotting modules."""

from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    FixationPSTHUnitPlotSettings,
    plot_fixation_psth_units,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_example_grid import (
    FixationPSTHExampleGridPlotSettings,
    FixationPSTHExampleUnitSpec,
    parse_example_grid_unit_specs,
    plot_fixation_psth_example_grid,
)
from dal_monte_2022_analysis.ephys.plotting.period_psth import (
    PeriodPSTHUnitPlotSettings,
    plot_period_psth_units,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_venn import (
    FixationSelectivityVennPlotSettings,
    build_fixation_selectivity_venn_summaries,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_selectivity_triangular import (
    FixationThreeWayTriangularPlotSettings,
    plot_fixation_three_way_selectivity_triangular,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonPlotSettings,
    plot_fixation_three_way_region_comparison_heatmaps,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_preference_index_heatmap import (
    FixationPreferenceIndexHeatmapPlotSettings,
    plot_fixation_preference_index_heatmaps,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_neural_cross_correlation import (
    FixationNeuralCrossCorrelationPlotSettings,
    plot_cross_region_fixation_neural_cross_correlation_summaries,
    plot_fixation_neural_cross_correlation_summaries,
    plot_within_region_fixation_neural_cross_correlation_summaries,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_population_pca import (
    FixationPopulationPCAPlotSettings,
    plot_fixation_population_pca_explained_variance_bars,
    plot_fixation_population_pca_explained_variance_cumulative,
    plot_fixation_population_pca_trajectories,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_roi_vs_period_factorial import (
    FixationROIVsPeriodFactorialPlotSettings,
    plot_fixation_roi_vs_period_axis_geometry,
    plot_fixation_roi_vs_period_axis_space,
    plot_fixation_roi_vs_period_axis_violin,
    plot_fixation_roi_vs_period_cross_region_graph,
)

__all__ = [
    "FixationPSTHUnitPlotSettings",
    "plot_fixation_psth_units",
    "FixationPSTHExampleGridPlotSettings",
    "FixationPSTHExampleUnitSpec",
    "parse_example_grid_unit_specs",
    "plot_fixation_psth_example_grid",
    "PeriodPSTHUnitPlotSettings",
    "plot_period_psth_units",
    "FixationSelectivityVennPlotSettings",
    "build_fixation_selectivity_venn_summaries",
    "FixationThreeWayTriangularPlotSettings",
    "plot_fixation_three_way_selectivity_triangular",
    "FixationThreeWayRegionComparisonPlotSettings",
    "plot_fixation_three_way_region_comparison_heatmaps",
    "FixationPreferenceIndexHeatmapPlotSettings",
    "plot_fixation_preference_index_heatmaps",
    "FixationNeuralCrossCorrelationPlotSettings",
    "plot_fixation_neural_cross_correlation_summaries",
    "plot_within_region_fixation_neural_cross_correlation_summaries",
    "plot_cross_region_fixation_neural_cross_correlation_summaries",
    "FixationPopulationPCAPlotSettings",
    "plot_fixation_population_pca_trajectories",
    "plot_fixation_population_pca_explained_variance_bars",
    "plot_fixation_population_pca_explained_variance_cumulative",
    "FixationROIVsPeriodFactorialPlotSettings",
    "plot_fixation_roi_vs_period_axis_violin",
    "plot_fixation_roi_vs_period_cross_region_graph",
    "plot_fixation_roi_vs_period_axis_space",
    "plot_fixation_roi_vs_period_axis_geometry",
]
