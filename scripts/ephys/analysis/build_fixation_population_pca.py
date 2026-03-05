"""Build region-level fixation population PCA outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_DEFAULT_DATASET_CFG = _REPO_ROOT / "configs" / "dataset.yaml"
_DEFAULT_EPHYS_FIX_PSTH_CFG = _REPO_ROOT / "configs" / "ephys_fixation_psth.yaml"

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pca import (
    FixationPopulationPCASettings,
    run_fixation_population_pca_analysis,
)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def _resolve_window_ms(raw: object) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        start = float(raw[0])
        stop = float(raw[1])
    else:
        start, stop = -500.0, 500.0
    if start > stop:
        start, stop = stop, start
    return start, stop


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit region-level fixation population PCA for "
            "face_interactive, face_non_interactive, and object conditions. "
            "Per-unit mean PSTHs are loaded from precomputed average PSTH files by default."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_DEFAULT_DATASET_CFG))
    parser.add_argument("--ephys-fixation-psth-cfg", default=str(_DEFAULT_EPHYS_FIX_PSTH_CFG))
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window-ms-start", type=float, default=None)
    parser.add_argument("--window-ms-stop", type=float, default=None)
    parser.add_argument("--max-components", type=int, default=None)
    parser.add_argument("--min-units-per-region", type=int, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)

    window_start_ms, window_stop_ms = _resolve_window_ms(cfg.get("population_pca_window_ms", [-500.0, 500.0]))
    if args.window_ms_start is not None:
        window_start_ms = float(args.window_ms_start)
    if args.window_ms_stop is not None:
        window_stop_ms = float(args.window_ms_stop)
    if window_start_ms > window_stop_ms:
        window_start_ms, window_stop_ms = window_stop_ms, window_start_ms

    settings = FixationPopulationPCASettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("population_pca_trial_input_modality", cfg.get("trial_output_modality", "psth")),
        trial_input_filename=cfg.get("population_pca_trial_input_filename", cfg.get("trial_output_filename", "fixations.pkl")),
        prefer_trial_input=cfg.get("population_pca_prefer_trial_input", False),
        allow_trial_fallback=cfg.get("population_pca_allow_trial_fallback", True),
        input_subdir=cfg.get(
            "population_pca_input_subdir",
            cfg.get(
                "average_output_subdir",
                cfg.get(
                    "selective_index_average_output_subdir",
                    "ephys/psth/fixation_psth_averages",
                ),
            ),
        ),
        input_filename=cfg.get(
            "population_pca_input_filename_split",
            cfg.get(
                "population_pca_input_filename",
                cfg.get(
                    "average_output_filename_split",
                    cfg.get(
                        "average_output_filename",
                        cfg.get(
                            "selective_index_average_output_filename_split",
                            "fixations.pkl",
                        ),
                    ),
                ),
            ),
        ),
        object_input_subdir=cfg.get(
            "population_pca_object_input_subdir",
            cfg.get("average_output_subdir_unsplit"),
        ),
        object_input_filename=cfg.get(
            "population_pca_object_input_filename",
            cfg.get(
                "average_output_filename_unsplit",
                cfg.get("selective_index_average_output_filename_unsplit"),
            ),
        ),
        output_subdir=cfg.get("population_pca_output_subdir", "ephys/psth/fixation_population_pca"),
        summary_filename=cfg.get("population_pca_summary_filename", "pca_fit_summary.csv"),
        timecourse_filename=cfg.get("population_pca_timecourse_filename", "concatenated_pc_timecourses.csv"),
        explained_variance_filename=cfg.get("population_pca_explained_variance_filename", "cross_condition_explained_variance.csv"),
        unit_inventory_filename=cfg.get("population_pca_unit_inventory_filename", "region_unit_inventory.csv"),
        output_pickle_filename=cfg.get("population_pca_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        window_start_ms=window_start_ms,
        window_stop_ms=window_stop_ms,
        max_components=cfg.get("population_pca_max_components", 50),
        min_units_per_region=cfg.get("population_pca_min_units_per_region", 3),
        require_all_conditions=cfg.get("population_pca_require_all_conditions", True),
        require_face_interactive_state=cfg.get("population_pca_require_face_interactive_state", True),
        smooth_before_average=cfg.get("population_pca_smooth_before_average", cfg.get("smooth_before_average", True)),
        smoothing_sigma_ms=cfg.get("population_pca_smoothing_sigma_ms", cfg.get("smoothing_sigma_ms", 20.0)),
        verbose_logging=cfg.get("population_pca_verbose_logging", True),
        use_parallel=cfg.get("population_pca_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s_fallback=cfg.get("window_pre_s", 1.0),
        window_post_s_fallback=cfg.get("window_post_s", 1.0),
    )

    if args.max_components is not None:
        settings.max_components = int(args.max_components)
    if args.min_units_per_region is not None:
        settings.min_units_per_region = int(args.min_units_per_region)
    if args.no_parallel:
        settings.use_parallel = False

    result = run_fixation_population_pca_analysis(
        settings,
        dates=args.date,
        regions=args.region,
    )
    fit_df = result.get("fit_summary")
    time_df = result.get("concatenated_timecourses")
    explained_df = result.get("cross_condition_explained_variance")
    unit_df = result.get("unit_inventory")

    n_fit_rows = 0 if fit_df is None else int(len(fit_df))
    n_regions = 0 if fit_df is None or fit_df.empty else int(fit_df["region"].astype(str).nunique())
    n_units = 0 if unit_df is None or unit_df.empty else int(unit_df["unit_key"].astype(str).nunique())
    n_time_rows = 0 if time_df is None else int(len(time_df))
    n_explained_rows = 0 if explained_df is None else int(len(explained_df))

    print(
        "[analysis] config paths: "
        f"dataset={dataset_cfg_path}, "
        f"ephys_fixation_psth={ephys_fix_psth_cfg_path}"
    )
    print(
        "[analysis] population PCA settings: "
        f"window_ms=[{settings.window_start_ms}, {settings.window_stop_ms}], "
        f"min_units_per_region={settings.min_units_per_region}, "
        f"max_components={settings.max_components}, "
        f"fixation_types={list(settings.conditions)}, "
        f"region_filter={args.region if args.region else 'all'}, "
        f"verbose_logging={settings.verbose_logging}"
    )
    print(
        "[analysis] population PCA input: "
        f"trial_modality={settings.trial_input_modality}, "
        f"prefer_trial_input={settings.prefer_trial_input}, "
        f"average_subdir={settings.input_subdir}, "
        f"object_average_subdir={settings.object_input_subdir}, "
        "dimension_reduction=units_to_pcs, "
        f"smooth_before_average={settings.smooth_before_average}, "
        f"smoothing_sigma_ms={settings.smoothing_sigma_ms}"
    )
    print(f"[analysis] fit summary rows: {n_fit_rows}")
    print(f"[analysis] analyzed regions: {n_regions}")
    print(f"[analysis] common units in outputs: {n_units}")
    print(f"[analysis] concatenated PC timecourse rows: {n_time_rows}")
    print(f"[analysis] cross-condition explained-variance rows: {n_explained_rows}")
    print("[analysis] wrote population PCA outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
