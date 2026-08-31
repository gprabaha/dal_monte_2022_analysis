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
from dal_monte_2022_analysis.ephys.analysis.fixation_neural_cross_correlation_pair_condition_means import (
    FixationNeuralCrossCorrelationPairConditionMeanSettings,
    build_fixation_neural_cross_correlation_pair_condition_mean_settings_from_config,
    run_cross_region_fixation_neural_cross_correlation_pair_condition_means,
    run_within_region_fixation_neural_cross_correlation_pair_condition_means,
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
from dal_monte_2022_analysis.ephys.analysis.fixation_condition_dominance import (
    FixationConditionDominanceSettings,
    run_fixation_condition_dominance_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pca import (
    FixationPopulationPCASettings,
    run_fixation_population_pca_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pc_subspace import (
    RegionPopulation,
    PopulationPCAFit,
    build_pairwise_subspace_table,
    build_region_subspace_summary,
    fit_all_scopes,
    fit_population_pca,
    load_region_populations,
    resolve_shared_n_components,
    verify_pca_identities,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_pair_spike_coordination import (
    PairSpikeCoordinationSettings,
    SessionSpikeTrains,
    build_group_z_traces,
    build_pair_spike_coordination_settings_from_config,
    build_pair_inventory,
    build_session_pair_table,
    build_zero_lag_diagnostics,
    compare_conditions,
    compute_condition_coordination,
    load_pair_coordination,
    load_session_spike_trains,
    run_pair_spike_coordination_build,
    summarize_coordination,
    test_against_null,
    verify_null_identities,
    verify_null_sensitivity,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_psth_variability import (
    FixationPSTHVariabilitySettings,
    run_fixation_psth_variability_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_roi_vs_period_factorial import (
    FixationROIVsPeriodFactorialSettings,
    run_fixation_roi_vs_period_factorial_analysis,
)
from dal_monte_2022_analysis.ephys.analysis.fixation_temporal_specificity import (
    METRIC_AXES,
    METRIC_LABELS,
    METRIC_NAMES,
    METRIC_SPECS,
    FixationTemporalSpecificitySettings,
    run_fixation_temporal_specificity_analysis,
)

__all__ = [
    "METRIC_AXES",
    "METRIC_LABELS",
    "METRIC_NAMES",
    "METRIC_SPECS",
    "FixationTemporalSpecificitySettings",
    "run_fixation_temporal_specificity_analysis",
    "FixationPSTHSelectivitySettings",
    "run_fixation_selectivity_analysis",
    "FixationPSTHPreferenceIndexSettings",
    "run_fixation_preference_index_analysis",
    "FixationThreeWayRegionComparisonSettings",
    "run_fixation_three_way_region_comparison",
    "FixationConditionDominanceSettings",
    "run_fixation_condition_dominance_analysis",
    "FixationPopulationPCASettings",
    "run_fixation_population_pca_analysis",
    "RegionPopulation",
    "PopulationPCAFit",
    "build_pairwise_subspace_table",
    "build_region_subspace_summary",
    "fit_all_scopes",
    "fit_population_pca",
    "load_region_populations",
    "resolve_shared_n_components",
    "verify_pca_identities",
    "PairSpikeCoordinationSettings",
    "SessionSpikeTrains",
    "build_group_z_traces",
    "build_pair_spike_coordination_settings_from_config",
    "build_pair_inventory",
    "build_session_pair_table",
    "build_zero_lag_diagnostics",
    "compare_conditions",
    "compute_condition_coordination",
    "load_pair_coordination",
    "load_session_spike_trains",
    "run_pair_spike_coordination_build",
    "summarize_coordination",
    "test_against_null",
    "verify_null_identities",
    "verify_null_sensitivity",
    "FixationPSTHVariabilitySettings",
    "run_fixation_psth_variability_analysis",
    "FixationROIVsPeriodFactorialSettings",
    "run_fixation_roi_vs_period_factorial_analysis",
    "WITHIN_ANALYSIS_KIND",
    "CROSS_ANALYSIS_KIND",
    "FixationNeuralCrossCorrelationSettings",
    "FixationNeuralCrossCorrelationPlotAggregationSettings",
    "run_within_region_fixation_neural_cross_correlation",
    "run_cross_region_fixation_neural_cross_correlation",
    "build_fixation_neural_cross_correlation_plot_payload",
    "build_within_region_fixation_neural_cross_correlation_plot_payload",
    "build_cross_region_fixation_neural_cross_correlation_plot_payload",
    "FixationNeuralCrossCorrelationPairConditionMeanSettings",
    "build_fixation_neural_cross_correlation_pair_condition_mean_settings_from_config",
    "run_within_region_fixation_neural_cross_correlation_pair_condition_means",
    "run_cross_region_fixation_neural_cross_correlation_pair_condition_means",
]
