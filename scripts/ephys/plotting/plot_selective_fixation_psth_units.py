"""Plot selective-unit fixation PSTHs (batch PNG + example PDF with sig ticks)."""

import argparse
from pathlib import Path

import pandas as pd

from dal_monte_2022_analysis.config.load import (
    load_dataset_config,
    load_ephys_fixation_psth_config,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_psth import (
    DEFAULT_CONDITION_COLORS,
    FixationPSTHUnitPlotSettings,
    plot_fixation_psth_units,
)


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    token = str(val).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _normalize_date_str(val) -> str:
    if val is None:
        return ""
    token = str(val).strip()
    if not token:
        return ""
    if token.endswith(".0"):
        token = token[:-2]
    if token.isdigit():
        return token.zfill(8)
    try:
        intval = int(float(token))
        return str(intval).zfill(8)
    except Exception:
        return token


def _load_selective_units_df(dataset_cfg_path: str, cfg: dict) -> pd.DataFrame:
    ds_cfg = load_dataset_config(dataset_cfg_path)
    out_root = Path(ds_cfg["analysis_output_root"]) / cfg.get(
        "selective_output_subdir",
        "ephys/psth/fixation_psth_selectivity",
    )
    unit_filename = cfg.get("selective_unit_summary_filename", "unit_selectivity.csv")
    unit_filename = unit_filename if str(unit_filename).endswith(".csv") else f"{unit_filename}.csv"
    unit_path = out_root / unit_filename
    if not unit_path.exists():
        raise FileNotFoundError(f"Selective unit summary not found: {unit_path}")

    df = pd.read_csv(unit_path)
    if df.empty:
        return df
    required = {"date", "unit_uuid", "is_selective_unit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Selective unit summary missing columns: {sorted(missing)}")

    df["date"] = df["date"].map(_normalize_date_str)
    df["unit_uuid"] = df["unit_uuid"].astype(str).map(lambda t: t.strip())
    df["is_selective_unit"] = df["is_selective_unit"].map(_as_bool)
    if "unit_key" not in df.columns:
        df["unit_key"] = df["date"].astype(str) + "|" + df["unit_uuid"].astype(str)
    else:
        df["unit_key"] = (
            df["unit_key"]
            .astype(str)
            .map(lambda t: t.strip())
        )
    return df


def _base_plot_settings(dataset_cfg_path: str, plotting_cfg_path: str, cfg: dict) -> FixationPSTHUnitPlotSettings:
    return FixationPSTHUnitPlotSettings(
        cfg_path=dataset_cfg_path,
        plotting_cfg_path=plotting_cfg_path,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get("plot_output_subdir", "ephys/psth/fixation_psth_unit_plots"),
        output_extension=cfg.get("plot_output_extension", "pdf"),
        output_dpi=cfg.get("plot_output_dpi", 220),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        use_parallel=cfg.get("plot_use_parallel", True),
        parallelize_units=cfg.get("plot_parallelize_units", True),
        unit_parallel_min_units=cfg.get("plot_unit_parallel_min_units", 2),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        max_trials_per_condition=cfg.get("plot_max_trials_per_condition", 300),
        random_seed=cfg.get("plot_random_seed", 42),
        condition_colors=cfg.get("plot_condition_colors", DEFAULT_CONDITION_COLORS),
        smooth_before_average=cfg.get("plot_smooth_before_average", True),
        smoothing_sigma_ms=cfg.get("plot_smoothing_sigma_ms", 20.0),
        raster_jitter_within_bin=cfg.get("plot_raster_jitter_within_bin", True),
        raster_linelength=cfg.get("plot_raster_linelength", 1.0),
        raster_linewidth=cfg.get("plot_raster_linewidth", 2.0),
        raster_alpha=cfg.get("plot_raster_alpha", 1.0),
        raster_darkening_factor=cfg.get("plot_raster_darkening_factor", 0.65),
        raster_show_condition_background=cfg.get("plot_raster_show_condition_background", False),
        panel_raster_height_ratio=cfg.get("plot_panel_raster_height_ratio", 1.2),
        panel_rate_height_ratio=cfg.get("plot_panel_rate_height_ratio", 2.0),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s=cfg.get("window_pre_s", 1.0),
        window_post_s=cfg.get("window_post_s", 1.0),
    )


def _build_batch_selective_plots(args, cfg: dict, selective_df: pd.DataFrame) -> int:
    sel = selective_df.loc[selective_df["is_selective_unit"]].copy()
    if args.date:
        target_date = _normalize_date_str(args.date)
        sel = sel.loc[sel["date"].astype(str) == target_date].copy()
    if args.unit_uuid:
        allowed = {str(u) for u in args.unit_uuid}
        sel = sel.loc[sel["unit_uuid"].astype(str).isin(allowed)].copy()

    if sel.empty:
        print("[plot] no selective units matched filters for batch plotting")
        return 0

    settings = _base_plot_settings(args.dataset_cfg, args.plotting_cfg, cfg)
    settings.output_subdir = cfg.get(
        "selective_plot_output_subdir",
        "ephys/psth/fixation_psth_selective_unit_plots",
    )
    settings.output_extension = cfg.get("selective_plot_output_extension", "png")
    settings.example_units_subfolder = None
    settings.show_significance_ticks = False

    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True

    out_paths = plot_fixation_psth_units(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_keys=sel["unit_key"].astype(str).tolist(),
    )
    print(f"[plot] wrote {len(out_paths)} selective-unit PNG plot(s)")
    if out_paths:
        print(f"[plot] first output: {out_paths[0]}")
    return len(out_paths)


def _build_example_plot(args, cfg: dict, selective_df: pd.DataFrame) -> int:
    target_uuid = str(args.example_unit_uuid).strip()
    sel = selective_df.loc[selective_df["is_selective_unit"]].copy()
    sel = sel.loc[sel["unit_uuid"].astype(str) == target_uuid].copy()
    if args.example_date:
        target_date = _normalize_date_str(args.example_date)
        sel = sel.loc[sel["date"].astype(str) == target_date].copy()

    if sel.empty:
        print("[plot] requested example unit is not in selective unit summary")
        return 0
    if len(sel) > 1 and not args.example_date:
        candidate_dates = sorted(sel["date"].astype(str).unique().tolist())
        print(
            "[plot] multiple selective entries found for this UUID across dates. "
            f"Re-run with --example-date. candidate dates: {candidate_dates}"
        )
        return 0

    settings = _base_plot_settings(args.dataset_cfg, args.plotting_cfg, cfg)
    settings.output_subdir = cfg.get(
        "selective_plot_output_subdir",
        "ephys/psth/fixation_psth_selective_unit_plots",
    )
    settings.output_extension = cfg.get("selective_example_output_extension", "pdf")
    settings.example_units_subfolder = cfg.get("selective_example_subfolder", "example units")
    settings.show_significance_ticks = True
    settings.significance_alpha = cfg.get("selective_example_significance_alpha", cfg.get("selective_alpha", 0.05))
    settings.significance_test = cfg.get("selective_example_significance_test", cfg.get("selective_test", "welch_ttest"))
    settings.significance_min_trials_per_condition = cfg.get(
        "selective_example_significance_min_trials_per_condition",
        cfg.get("selective_min_trials_per_condition", 2),
    )
    settings.use_parallel = False
    settings.parallelize_units = False
    settings.test_single = False

    unit_keys = sel["unit_key"].astype(str).tolist()
    target_dates = sorted(sel["date"].astype(str).unique().tolist())
    out_paths = plot_fixation_psth_units(
        settings,
        dates=target_dates,
        sessions=[args.session] if args.session else None,
        unit_keys=unit_keys,
    )
    if not out_paths:
        print(
            "[plot] no example plot written. "
            f"resolved_dates={target_dates}, unit_keys={unit_keys}"
        )
    print(f"[plot] wrote {len(out_paths)} example selective-unit PDF plot(s)")
    if out_paths:
        print(f"[plot] output: {out_paths[0]}")
    return len(out_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selective-unit fixation PSTHs. Default: batch PNG for all selective units. "
            "Example mode: single-unit PDF with per-bin significance ticks."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--example-unit-uuid", default=None)
    parser.add_argument("--example-date", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_ephys_fixation_psth_config(args.ephys_fixation_psth_cfg)
    selective_df = _load_selective_units_df(args.dataset_cfg, cfg)
    if selective_df.empty:
        print("[plot] selective unit summary is empty")
        return

    if args.example_unit_uuid:
        _build_example_plot(args, cfg, selective_df)
        return
    _build_batch_selective_plots(args, cfg, selective_df)


if __name__ == "__main__":
    main()
