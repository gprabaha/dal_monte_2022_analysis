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
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_phasic_tonic_example_grid import (
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS,
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_LABELS,
    DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES,
    normalize_example_response_style,
    parse_phasic_tonic_example_grid_unit_specs,
    plot_fixation_psth_phasic_tonic_example_grid,
)
from dal_monte_2022_analysis.ephys.plotting.period_psth import (
    PeriodPSTHUnitPlotSettings,
    plot_period_psth_units,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_venn import (
    FixationSelectivityVennPlotSettings,
    build_fixation_selectivity_venn_summaries,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_comparison_group import (
    FixationSelectivityComparisonGroupPlotSettings,
    plot_fixation_selectivity_comparison_group_summaries,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_selectivity_triangular import (
    FixationThreeWayTriangularPlotSettings,
    plot_fixation_three_way_selectivity_triangular,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_three_way_region_comparison import (
    FixationThreeWayRegionComparisonPlotSettings,
    plot_fixation_three_way_region_comparison_heatmaps,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_condition_dominance import (
    FixationConditionDominancePlotSettings,
    plot_fixation_condition_dominance_by_region,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_peakiness import (
    FixationPeakinessPlotSettings,
    plot_fixation_peakiness_by_region,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_peakiness_condition_comparison import (
    FixationPeakinessConditionComparisonPlotSettings,
    plot_fixation_peakiness_condition_comparison,
    plot_fixation_peakiness_condition_comparison_by_region,
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
from dal_monte_2022_analysis.ephys.plotting.fixation_neural_cross_correlation_pair_condition_means import (
    FixationNeuralCrossCorrelationPairConditionMeanPlotSettings,
    plot_cross_region_fixation_neural_cross_correlation_pair_condition_mean_summaries,
    plot_fixation_neural_cross_correlation_pair_condition_mean_summaries,
    plot_within_region_fixation_neural_cross_correlation_pair_condition_mean_summaries,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_population_pca import (
    FixationPopulationPCAPlotSettings,
    plot_fixation_population_pca_explained_variance_bars,
    plot_fixation_population_pca_explained_variance_cumulative,
    plot_fixation_population_pca_pairwise_geometry_violins,
    plot_fixation_population_pca_trajectories,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_population_pc_subspace import (
    optimize_view_angle,
    plot_alignment_matrix,
    plot_cross_condition_variance_curves,
    plot_cumulative_variance,
    plot_pc_plane_projections,
    plot_pc_trajectories_3d,
    plot_principal_angle_spectra,
    plot_subspace_distance_map,
    plot_time_resolved_separation,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_pair_spatial_decay import (
    SpatialDecayPlotSettings,
    plot_condition_by_separation,
    plot_confound_schematic,
    plot_decay_by_condition,
    plot_decay_curves,
    plot_fit_parameters,
    plot_method_schematic,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_pair_spike_coordination import (
    PairCoordinationPlotSettings,
    plot_condition_contrasts,
    plot_null_corrected_grid,
    plot_observed_and_null_grid,
    plot_zero_lag_diagnostics,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_mrnn import (
    FixationMRNNDiagnosticPlotSettings,
    plot_fixation_mrnn_activation_pc_timeseries,
    plot_fixation_mrnn_activation_trajectories_3d,
    plot_fixation_mrnn_average_current_influence_bars,
    plot_fixation_mrnn_average_current_influence_pies,
    plot_fixation_mrnn_current_influence,
    plot_fixation_mrnn_flow_fields_at_time,
    plot_fixation_mrnn_signal_evolution,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_variability import (
    FixationPSTHVariabilityPlotSettings,
    plot_fixation_psth_variability_violins,
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
    "DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS",
    "DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_LABELS",
    "DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES",
    "normalize_example_response_style",
    "parse_phasic_tonic_example_grid_unit_specs",
    "plot_fixation_psth_phasic_tonic_example_grid",
    "PeriodPSTHUnitPlotSettings",
    "plot_period_psth_units",
    "FixationSelectivityVennPlotSettings",
    "build_fixation_selectivity_venn_summaries",
    "FixationSelectivityComparisonGroupPlotSettings",
    "plot_fixation_selectivity_comparison_group_summaries",
    "FixationThreeWayTriangularPlotSettings",
    "plot_fixation_three_way_selectivity_triangular",
    "FixationThreeWayRegionComparisonPlotSettings",
    "plot_fixation_three_way_region_comparison_heatmaps",
    "FixationConditionDominancePlotSettings",
    "plot_fixation_condition_dominance_by_region",
    "FixationPeakinessPlotSettings",
    "plot_fixation_peakiness_by_region",
    "FixationPeakinessConditionComparisonPlotSettings",
    "plot_fixation_peakiness_condition_comparison",
    "plot_fixation_peakiness_condition_comparison_by_region",
    "FixationPreferenceIndexHeatmapPlotSettings",
    "plot_fixation_preference_index_heatmaps",
    "FixationNeuralCrossCorrelationPlotSettings",
    "plot_fixation_neural_cross_correlation_summaries",
    "plot_within_region_fixation_neural_cross_correlation_summaries",
    "plot_cross_region_fixation_neural_cross_correlation_summaries",
    "FixationNeuralCrossCorrelationPairConditionMeanPlotSettings",
    "plot_fixation_neural_cross_correlation_pair_condition_mean_summaries",
    "plot_within_region_fixation_neural_cross_correlation_pair_condition_mean_summaries",
    "plot_cross_region_fixation_neural_cross_correlation_pair_condition_mean_summaries",
    "FixationPopulationPCAPlotSettings",
    "plot_fixation_population_pca_trajectories",
    "plot_fixation_population_pca_explained_variance_bars",
    "plot_fixation_population_pca_explained_variance_cumulative",
    "plot_fixation_population_pca_pairwise_geometry_violins",
    "optimize_view_angle",
    "plot_alignment_matrix",
    "plot_cross_condition_variance_curves",
    "plot_cumulative_variance",
    "plot_pc_plane_projections",
    "plot_pc_trajectories_3d",
    "plot_principal_angle_spectra",
    "plot_subspace_distance_map",
    "plot_time_resolved_separation",
    "SpatialDecayPlotSettings",
    "plot_condition_by_separation",
    "plot_confound_schematic",
    "plot_decay_by_condition",
    "plot_decay_curves",
    "plot_fit_parameters",
    "plot_method_schematic",
    "PairCoordinationPlotSettings",
    "plot_condition_contrasts",
    "plot_null_corrected_grid",
    "plot_observed_and_null_grid",
    "plot_zero_lag_diagnostics",
    "FixationMRNNDiagnosticPlotSettings",
    "plot_fixation_mrnn_activation_pc_timeseries",
    "plot_fixation_mrnn_activation_trajectories_3d",
    "plot_fixation_mrnn_average_current_influence_bars",
    "plot_fixation_mrnn_average_current_influence_pies",
    "plot_fixation_mrnn_current_influence",
    "plot_fixation_mrnn_flow_fields_at_time",
    "plot_fixation_mrnn_signal_evolution",
    "FixationPSTHVariabilityPlotSettings",
    "plot_fixation_psth_variability_violins",
    "FixationROIVsPeriodFactorialPlotSettings",
    "plot_fixation_roi_vs_period_axis_violin",
    "plot_fixation_roi_vs_period_cross_region_graph",
    "plot_fixation_roi_vs_period_axis_space",
    "plot_fixation_roi_vs_period_axis_geometry",
]
