"""Data assembly for the thesis single-unit figures.

Keeps every join, filter and trial-store read out of the notebooks: each
function returns a plain table or a list of ready-to-draw panel specs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_peakiness import (
    FixationPeakinessSettings,
    decompose_dominant_peak_prominence,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    DEFAULT_CONDITION_COLORS,
    FixationPSTHUnitPlotSettings,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth_example_grid import (
    FixationPSTHExampleUnitSpec,
    _load_trials_by_date,
    _resolve_cell,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_common import (
    DPP_COLUMN,
    REGION_ORDER,
    bare_unit_uuid,
    region_label,
)
from dal_monte_2022_analysis.ephys.plotting.thesis_single_unit import ExampleUnitPanelSpec
from dal_monte_2022_analysis.runtime.io.analysis_index import build_analysis_output_dir
from dal_monte_2022_analysis.runtime.io.processed_data import load_pickle_path

PRIMARY_COMPARISON_LABEL = "three_condition_core"


def resolve_psth_analysis_root(dataset_cfg: Mapping) -> Path:
    """Root directory holding every fixation-PSTH analysis product."""
    return build_analysis_output_dir(dict(dataset_cfg), "ephys/psth")


def load_thesis_unit_table(
    analysis_root: Path,
    *,
    comparison_label: str = PRIMARY_COMPARISON_LABEL,
) -> pd.DataFrame:
    """One row per recorded unit, joining selectivity, preference and DPP.

    ``is_selective`` is the FDR-corrected call across all pair x window tests
    within a unit; the uncorrected call is kept as ``is_selective_raw`` so the
    chapter can state how much of the yield survives correction.
    """
    selectivity = pd.read_csv(analysis_root / "fixation_psth_selectivity" / "unit_selectivity.csv")
    selectivity = selectivity.loc[
        selectivity["comparison_label"].astype(str) == str(comparison_label)
    ]
    units = selectivity.loc[
        :,
        [
            "unit_key",
            "date",
            "unit_uuid",
            "region",
            "spike_channel",
            "recorded_agent",
            "is_selective_unit_corrected",
            "is_selective_unit_raw",
            "n_selective_pairs_corrected",
            "selective_pairs_corrected",
        ],
    ].rename(
        columns={
            "is_selective_unit_corrected": "is_selective",
            "is_selective_unit_raw": "is_selective_raw",
            "n_selective_pairs_corrected": "n_selective_pairs",
            "selective_pairs_corrected": "selective_pairs",
        }
    )

    dominance = pd.read_csv(
        analysis_root / "fixation_condition_dominance" / "unit_condition_dominance.csv"
    )
    units = units.merge(
        dominance.loc[:, ["unit_key", "dominant_condition", "dominance_status"]],
        on="unit_key",
        how="left",
    )

    peakiness = pd.read_csv(analysis_root / "fixation_peakiness" / "unit_peakiness.csv")
    peakiness_columns = [
        "unit_key",
        DPP_COLUMN,
        "best_condition",
        "best_peak_latency_ms",
        "best_peak_prominence",
        "best_peak_competition_ratio",
    ]
    for condition in ("face_interactive", "face_non_interactive", "object"):
        column = f"{condition}_{DPP_COLUMN}"
        if column in peakiness.columns:
            peakiness_columns.append(column)
        mean_column = f"{condition}_mean_fr_hz"
        if mean_column in peakiness.columns:
            peakiness_columns.append(mean_column)
    units = units.merge(peakiness.loc[:, peakiness_columns], on="unit_key", how="left")

    units["region"] = units["region"].astype(str).str.strip().str.lower()
    units["uuid"] = units["unit_uuid"].map(bare_unit_uuid)
    units["region_label"] = units["region"].map(region_label)
    units["is_selective"] = units["is_selective"].astype(bool)
    units["is_selective_raw"] = units["is_selective_raw"].astype(bool)
    units["dpp_percentile"] = units.groupby("region")[DPP_COLUMN].rank(pct=True)

    if not units["unit_key"].is_unique:
        raise ValueError("unit_key is not unique after joining the thesis unit tables.")
    return units


def load_pair_selectivity(
    analysis_root: Path,
    *,
    comparison_label: str = PRIMARY_COMPARISON_LABEL,
) -> pd.DataFrame:
    """Per unit x pair selectivity rows for the primary comparison group."""
    path = analysis_root / "fixation_psth_selectivity" / f"pair_selectivity__{comparison_label}.csv"
    if not path.exists():
        path = analysis_root / "fixation_psth_selectivity" / "pair_selectivity.csv"
    frame = pd.read_csv(path)
    if "comparison_label" in frame.columns:
        frame = frame.loc[frame["comparison_label"].astype(str) == str(comparison_label)]
    frame["region"] = frame["region"].astype(str).str.strip().str.lower()
    return frame


def load_condition_variability(analysis_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-unit CV/SD table and the stored within-region paired tests."""
    directory = analysis_root / "fixation_psth_variability"
    variability = pd.read_csv(directory / "unit_condition_variability.csv")
    stats_table = pd.read_csv(directory / "within_region_condition_variability_stats.csv")
    for frame in (variability, stats_table):
        frame["region"] = frame["region"].astype(str).str.strip().str.lower()
    return variability, stats_table


def load_trace_metric_table(
    analysis_root: Path,
    units: pd.DataFrame,
    *,
    selective_only: bool = True,
) -> pd.DataFrame:
    """Trace-shape metrics joined onto the unit table.

    Defaults to the fixation-category-modulated subpopulation: a width or
    raggedness statistic computed on an unmodulated trace describes the noise
    floor, not a response.
    """
    metrics = pd.read_csv(
        analysis_root / "fixation_temporal_specificity" / "unit_temporal_specificity.csv"
    )
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"date", "unit_uuid", "region", "spike_channel", "recorded_agent",
                          "recorded_monkey", "area"}
    ]
    merged = units.merge(metrics.loc[:, metric_columns], on="unit_key", how="inner")
    if selective_only:
        merged = merged.loc[merged["is_selective"]].copy()
    return merged


def load_trace_shape_table(
    analysis_root: Path,
    units: pd.DataFrame,
    *,
    condition: str = "dominant",
) -> pd.DataFrame:
    """Response duration and Peak Isolation Index per unit.

    The chapter describes each unit with two orthogonal, separately drawable
    quantities rather than one composite:

    ``response_duration_ms``
        Time the excess response stays at or above half its peak (FWHM). Being a
        width, it is amplitude-invariant and therefore far less sensitive to
        trial count than any prominence-based statistic.
    ``peak_isolation``
        ``1 - P2/P1``: 1 when the dominant peak has no rival at least 250 ms
        away, 0 when an equally strong second peak exists. This is the isolation
        term of the old composite with the amplitude factor removed -- the
        composite ``peakiness_score`` correlates 0.99 with ``P1`` alone, so it
        measured peak height while being described as measuring isolation.

    ``condition="dominant"`` scores each unit on the fixation category it fires
    most for, which is the trace the chapter's example panels show.
    """
    peakiness = pd.read_csv(
        analysis_root / "fixation_peakiness" / "unit_condition_peakiness.csv"
    )
    specificity = pd.read_csv(
        analysis_root / "fixation_temporal_specificity" / "unit_condition_temporal_specificity.csv"
    )
    merged = peakiness.loc[
        :,
        [
            "unit_key",
            "condition",
            "n_trials",
            "mean_fr_hz",
            "best_peak_prominence",
            "second_peak_prominence",
            "competition_ratio",
            "best_peak_latency_ms",
            "peakiness_score",
        ],
    ].merge(
        specificity.loc[
            :,
            ["unit_key", "condition", "fwhm_frac", "sustained_frac", "roughness",
             "peak_z", "n_prominent_peaks", "window_duration_ms"],
        ],
        on=["unit_key", "condition"],
        how="inner",
    )
    merged["peak_isolation"] = 1.0 - merged["competition_ratio"].astype(float)
    merged["response_duration_ms"] = (
        merged["fwhm_frac"].astype(float) * merged["window_duration_ms"].astype(float)
    )

    keep = units.loc[
        :, ["unit_key", "region", "uuid", "date", "is_selective", "dominant_condition"]
    ]
    merged = merged.merge(keep, on="unit_key", how="inner")
    if condition == "dominant":
        merged = merged.loc[
            merged["condition"].astype(str) == merged["dominant_condition"].astype(str)
        ].copy()
    elif condition != "all":
        merged = merged.loc[merged["condition"].astype(str) == str(condition)].copy()
    return merged.reset_index(drop=True)


def load_trial_matched_cv(analysis_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trial-count-matched CV table, its within-region tests, and the inflation curve."""
    directory = analysis_root / "fixation_cv_trial_matched"
    unit_cv = pd.read_csv(directory / "unit_condition_cv_trial_matched.csv")
    stats_table = pd.read_csv(directory / "within_region_cv_trial_matched_stats.csv")
    inflation = pd.read_csv(directory / "cv_trial_count_inflation.csv")
    for frame in (unit_cv, stats_table, inflation):
        if "region" in frame.columns:
            frame["region"] = frame["region"].astype(str).str.strip().str.lower()
    return unit_cv, stats_table, inflation


def load_condition_traces(analysis_root: Path) -> pd.DataFrame:
    """Windowed condition-average traces, used for the DPP decomposition."""
    return load_pickle_path(
        analysis_root / "fixation_temporal_specificity" / "unit_condition_traces.pkl"
    )


def build_exemplar_table(
    units: pd.DataFrame,
    exemplar_specs: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Attach unit-table columns to a ``{region: {style: uuid}}`` exemplar map."""
    rows = []
    for region, styles in exemplar_specs.items():
        region_key = str(region).strip().lower()
        for style, uuid in styles.items():
            match = units.loc[
                (units["region"] == region_key) & (units["uuid"].astype(str) == str(uuid))
            ]
            if match.empty:
                raise ValueError(f"Exemplar unit not found: region={region}, uuid={uuid}")
            row = match.iloc[0].to_dict()
            row["style"] = str(style)
            rows.append(row)
    return pd.DataFrame(rows)


def select_metric_driven_exemplars(
    units: pd.DataFrame,
    *,
    regions: Sequence[str] = REGION_ORDER,
    score_column: str = DPP_COLUMN,
    require_selective: bool = True,
    min_mean_fr_hz: float = 3.0,
    max_mean_fr_hz: float = 30.0,
    mean_fr_column: str = "face_interactive_mean_fr_hz",
) -> pd.DataFrame:
    """Highest- and lowest-DPP unit per region, as data-driven alternates.

    The firing-rate window keeps the picks plottable: a sub-1 Hz unit has an
    uninformative raster and a very high-rate unit compresses the PSTH axis, and
    neither failure says anything about the score.
    """
    rows = []
    for region in regions:
        pool = units.loc[units["region"].astype(str) == str(region)].copy()
        if require_selective:
            pool = pool.loc[pool["is_selective"]]
        if mean_fr_column in pool.columns:
            rate = pd.to_numeric(pool[mean_fr_column], errors="coerce")
            pool = pool.loc[rate.between(min_mean_fr_hz, max_mean_fr_hz)]
        pool = pool.loc[pd.to_numeric(pool[score_column], errors="coerce").notna()]
        if pool.empty:
            continue
        high = pool.nlargest(1, score_column).iloc[0].to_dict()
        high["style"] = "phasic"
        low = pool.nsmallest(1, score_column).iloc[0].to_dict()
        low["style"] = "tonic"
        rows.extend([high, low])
    return pd.DataFrame(rows)


def build_unit_plot_settings(
    dataset_cfg_path: str | Path,
    plotting_cfg_path: str | Path,
    psth_cfg: Mapping,
) -> FixationPSTHUnitPlotSettings:
    """Trial/average sources for one example-unit column."""
    return FixationPSTHUnitPlotSettings(
        cfg_path=str(dataset_cfg_path),
        plotting_cfg_path=str(plotting_cfg_path),
        trial_input_modality=psth_cfg.get("trial_output_modality", "psth"),
        trial_input_filename=psth_cfg.get("plot_trial_input_filename", "fixations_psth_10ms.pkl"),
        raster_trial_input_modality=psth_cfg.get("trial_output_modality", "psth"),
        raster_trial_input_filename=psth_cfg.get(
            "plot_raster_trial_input_filename", "fixations_spike_train_1ms.pkl"
        ),
        use_precomputed_average_traces=True,
        average_trace_input_subdir=psth_cfg.get(
            "plot_average_input_subdir", "ephys/psth/fixation_psth_averages"
        ),
        average_trace_input_filename=psth_cfg.get(
            "plot_average_input_filename_split",
            psth_cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
        ),
        average_trace_object_input_subdir=psth_cfg.get(
            "plot_average_object_input_subdir",
            psth_cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        average_trace_object_input_filename=psth_cfg.get(
            "plot_average_object_input_filename",
            psth_cfg.get("plot_average_input_filename_unsplit", "fixations_psth_10ms.pkl"),
        ),
        allow_trial_trace_fallback=False,
        interactive_label=psth_cfg.get("interactive_high_label", "interactive"),
        use_parallel=False,
        parallelize_units=False,
        max_trials_per_condition=psth_cfg.get("plot_max_trials_per_condition", 300),
        random_seed=psth_cfg.get("plot_random_seed", 42),
        condition_colors=psth_cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS),
        smooth_before_average=True,
        smoothing_sigma_ms=psth_cfg.get("plot_smoothing_sigma_ms", 20.0),
        bin_size_ms_fallback=psth_cfg.get("bin_size_ms", 10.0),
        window_pre_s=psth_cfg.get("window_pre_s", 1.0),
        window_post_s=psth_cfg.get("window_post_s", 1.0),
    )


def build_peakiness_settings(
    dataset_cfg_path: str | Path,
    psth_cfg: Mapping,
) -> FixationPeakinessSettings:
    """Peakiness settings matching the stored score, for decomposition."""
    return FixationPeakinessSettings(
        cfg_path=str(dataset_cfg_path),
        rate_normalization_mode=psth_cfg.get("peakiness_rate_normalization_mode", "sqrt_mean"),
        mean_rate_floor_hz=float(psth_cfg.get("peakiness_mean_rate_floor_hz", 0.5)),
        peak_distance_ms=float(psth_cfg.get("peakiness_peak_distance_ms", 30.0)),
        peak_prominence_floor=float(psth_cfg.get("peakiness_peak_prominence_floor", 0.0)),
        competition_penalty_lambda=float(
            psth_cfg.get("peakiness_competition_penalty_lambda", 0.5)
        ),
        competition_exclusion_window_ms=float(
            psth_cfg.get("peakiness_competition_exclusion_window_ms", 250.0)
        ),
        bin_size_ms_fallback=float(psth_cfg.get("bin_size_ms", 10.0)),
    )


def build_example_unit_panel_specs(
    exemplars: pd.DataFrame,
    *,
    style: str,
    unit_settings: FixationPSTHUnitPlotSettings,
    peakiness_settings: FixationPeakinessSettings,
    condition_traces: pd.DataFrame,
    regions: Sequence[str] = REGION_ORDER,
    schematic_region: Optional[str] = None,
    schematic_condition: str = "face_interactive",
) -> list[ExampleUnitPanelSpec]:
    """Load trials for one exemplar row and package them for plotting.

    Only the dates the exemplars actually come from are scanned, which keeps a
    four-column figure to a handful of trial-store reads instead of a full sweep.
    """
    selected = exemplars.loc[exemplars["style"].astype(str) == str(style)]
    ordered = [
        selected.loc[selected["region"].astype(str) == str(region)].iloc[0]
        for region in regions
        if not selected.loc[selected["region"].astype(str) == str(region)].empty
    ]
    if not ordered:
        raise ValueError(f"No exemplars found for style={style!r}.")

    dates = sorted({str(unit["date"]).zfill(8) for unit in ordered})
    by_date = _load_trials_by_date(unit_settings, date_filter=dates)

    specs: list[ExampleUnitPanelSpec] = []
    for unit in ordered:
        region = str(unit["region"])
        uuid = str(unit["uuid"])
        date = str(unit["date"]).zfill(8)
        cell, error = _resolve_cell(
            FixationPSTHExampleUnitSpec(
                region=region_label(region),
                preference=str(style),
                unit_uuid=uuid,
                date=date,
            ),
            by_date,
            unit_settings,
        )
        if cell is None:
            raise ValueError(f"Unable to resolve example unit {region} {uuid}: {error}")

        decomposition = None
        if schematic_region is not None and region == str(schematic_region):
            trace_rows = condition_traces.loc[
                (condition_traces["unit_key"].astype(str) == str(unit["unit_key"]))
                & (condition_traces["condition"].astype(str) == str(schematic_condition))
            ]
            if trace_rows.empty:
                raise ValueError(
                    f"No {schematic_condition} average trace stored for {region} {uuid}; "
                    "cannot build the DPP schematic."
                )
            trace_row = trace_rows.iloc[0]
            decomposition = decompose_dominant_peak_prominence(
                np.asarray(trace_row["trace_hz"], dtype=float),
                np.asarray(trace_row["bin_centers_s_rel"], dtype=float),
                peakiness_settings,
            )

        specs.append(
            ExampleUnitPanelSpec(
                region=region,
                unit_uuid=uuid,
                date=date,
                bin_centers_s=np.asarray(cell.bin_centers, dtype=float),
                payloads=cell.payloads,
                dpp_score=float(unit[DPP_COLUMN]),
                dpp_percentile=float(unit["dpp_percentile"]),
                decomposition=decomposition,
            )
        )
    return specs


def parse_config_exemplar_map(
    psth_cfg: Mapping,
    *,
    cfg_key: str = "phasic_tonic_example_grid_units",
) -> dict[str, dict[str, str]]:
    """Normalize the configured example-unit map to ``{region: {style: uuid}}``."""
    raw = psth_cfg.get(cfg_key, {}) or {}
    out: dict[str, dict[str, str]] = {}
    for region, styles in raw.items():
        region_key = str(region).strip().lower()
        entries: dict[str, str] = {}
        for style, value in (styles or {}).items():
            if isinstance(value, Mapping):
                uuid = value.get("unit_uuid")
            else:
                uuid = value
            if uuid is None:
                continue
            entries[str(style).strip().lower()] = bare_unit_uuid(uuid)
        if entries:
            out[region_key] = entries
    return out
