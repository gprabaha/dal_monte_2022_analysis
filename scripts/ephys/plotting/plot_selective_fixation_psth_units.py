"""Plot selective-unit fixation PSTHs (batch PNG + example PDF with sig ticks)."""

import argparse

import pandas as pd

from dal_monte_2022_analysis.config.load import load_config
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


def _normalize_ext_list(raw, *, fallback: tuple[str, ...]) -> list[str]:
    if raw is None:
        seq = list(fallback)
    elif isinstance(raw, (list, tuple)):
        seq = [str(v).strip() for v in raw]
    else:
        seq = [str(raw).strip()]
    cleaned: list[str] = []
    for ext in seq:
        if not ext:
            continue
        token = ext.lower()
        if token.startswith("."):
            token = token[1:]
        if token and token not in cleaned:
            cleaned.append(token)
    return cleaned or list(fallback)


def _normalize_float_list(raw, *, fallback):
    if raw is None:
        seq = list(fallback)
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        seq = [raw]
    out = []
    for item in seq:
        try:
            value = float(item)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    return out or list(fallback)


def _normalize_color_list(raw, *, fallback):
    if raw is None:
        seq = list(fallback)
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        seq = [raw]
    out = [str(item).strip() for item in seq if str(item).strip()]
    return out or list(fallback)


def _resolve_analysis_windows_s(cfg: dict) -> list[tuple[float, float]]:
    raw = cfg.get("plot_analysis_windows_ms")
    out: list[tuple[float, float]] = []
    if isinstance(raw, dict):
        raw = [raw.get(key) for key in ("pre_fix", "peri_fix", "post_fix")]
    if raw is None:
        selective_windows = cfg.get("selective_windows_ms")
        if isinstance(selective_windows, dict):
            raw = [selective_windows.get(key) for key in ("pre_fix", "peri_fix", "post_fix")]
    if isinstance(raw, (list, tuple)):
        for bounds in raw:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            try:
                start_s = float(bounds[0]) / 1000.0
                stop_s = float(bounds[1]) / 1000.0
            except Exception:
                continue
            if start_s > stop_s:
                start_s, stop_s = stop_s, start_s
            out.append((start_s, stop_s))
    return out or [(-0.5, 0.0), (-0.25, 0.25), (0.0, 0.5)]


def _load_selective_units_df(dataset_cfg_path: str, cfg: dict) -> pd.DataFrame:
    ds_cfg = load_config(dataset_cfg_path)
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
        trial_input_modality=cfg.get("plot_trial_input_modality", cfg.get("trial_output_modality", "psth")),
        trial_input_filename=cfg.get("plot_trial_input_filename", "fixations_psth_10ms.pkl"),
        raster_trial_input_modality=cfg.get(
            "plot_raster_trial_input_modality",
            cfg.get("plot_trial_input_modality", cfg.get("trial_output_modality", "psth")),
        ),
        raster_trial_input_filename=cfg.get("plot_raster_trial_input_filename", "fixations_spike_train_1ms.pkl"),
        use_precomputed_average_traces=cfg.get("plot_use_precomputed_average_traces", True),
        average_trace_input_subdir=cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        average_trace_input_filename=cfg.get(
            "plot_average_input_filename_split",
            cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl"),
        ),
        average_trace_object_input_subdir=cfg.get(
            "plot_average_object_input_subdir",
            cfg.get("plot_average_input_subdir", "ephys/psth/fixation_psth_averages"),
        ),
        average_trace_object_input_filename=cfg.get(
            "plot_average_object_input_filename",
            cfg.get("plot_average_input_filename_unsplit", cfg.get("plot_average_input_filename", "fixations_psth_10ms.pkl")),
        ),
        allow_trial_trace_fallback=cfg.get("plot_allow_trial_trace_fallback", True),
        segregate_selective_units=cfg.get("plot_segregate_selective_units", True),
        selectivity_input_subdir=cfg.get("plot_selectivity_input_subdir", cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity")),
        selectivity_unit_summary_filename=cfg.get("plot_selectivity_unit_summary_filename", cfg.get("selective_unit_summary_filename", "unit_selectivity.csv")),
        selective_unit_subfolder=cfg.get("plot_selective_unit_subfolder", "selective"),
        output_subdir=cfg.get("plot_output_subdir", "ephys/psth/fixation_psth_unit_plots_multiscale_5s"),
        output_extension=cfg.get("plot_output_extension", "pdf"),
        figure_size=cfg.get("plot_figsize"),
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
        display_half_windows_s=_normalize_float_list(
            cfg.get("plot_display_half_windows_s"),
            fallback=(5.0, 3.0, 1.0),
        ),
        show_analysis_window_overlays=cfg.get("plot_show_analysis_window_overlays", True),
        analysis_window_overlays_s=_resolve_analysis_windows_s(cfg),
        analysis_window_overlay_colors=_normalize_color_list(
            cfg.get("plot_analysis_window_colors"),
            fallback=("#bdbdbd", "#8f8f8f", "#636363"),
        ),
        analysis_window_overlay_linestyle=cfg.get("plot_analysis_window_linestyle", ":"),
        analysis_window_overlay_linewidth=cfg.get("plot_analysis_window_linewidth", 0.8),
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
        "ephys/psth/fixation_psth_unit_plots_multiscale_5s",
    )
    settings.output_extension = cfg.get("selective_plot_output_extension", "png")
    settings.selective_unit_subfolder = cfg.get(
        "plot_selective_unit_subfolder",
        cfg.get("selective_plot_region_subfolder", "selective"),
    )
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
        "ephys/psth/fixation_psth_unit_plots_multiscale_5s",
    )
    settings.selective_unit_subfolder = cfg.get(
        "plot_selective_unit_subfolder",
        cfg.get("selective_plot_region_subfolder", "selective"),
    )
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
    exts = _normalize_ext_list(
        cfg.get("selective_example_output_extensions", cfg.get("selective_example_output_extension")),
        fallback=("pdf", "png"),
    )
    all_out_paths = []
    for ext in exts:
        settings.output_extension = ext
        out_paths = plot_fixation_psth_units(
            settings,
            dates=target_dates,
            sessions=[args.session] if args.session else None,
            unit_keys=unit_keys,
        )
        all_out_paths.extend(out_paths)
        print(f"[plot] wrote {len(out_paths)} example selective-unit {ext.upper()} plot(s)")
        if out_paths:
            print(f"[plot] {ext} output: {out_paths[0]}")

    if not all_out_paths:
        print(
            "[plot] no example plot written. "
            f"resolved_dates={target_dates}, unit_keys={unit_keys}"
        )
    return len(all_out_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot multiscale selective-unit fixation PSTHs. Default: batch PNG for all "
            "selective units. Example mode: single-unit PDF with per-bin significance ticks."
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

    cfg = load_config(args.ephys_fixation_psth_cfg)
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
