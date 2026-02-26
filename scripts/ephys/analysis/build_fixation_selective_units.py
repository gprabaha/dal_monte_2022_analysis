"""Run fixation-pair selective-unit analysis from trial PSTH data."""

import argparse

from dal_monte_2022_analysis.config.load import load_ephys_fixation_psth_config
from dal_monte_2022_analysis.ephys.analysis.fixation_selectivity import (
    DEFAULT_SELECTIVITY_WINDOWS_MS,
    FixationPSTHSelectivitySettings,
    run_fixation_selectivity_analysis,
)


def _as_str_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def _normalize_windows_cfg(raw):
    if raw is None:
        return dict(DEFAULT_SELECTIVITY_WINDOWS_MS)
    if isinstance(raw, dict):
        out = {}
        for name, bounds in raw.items():
            if bounds is None or len(bounds) != 2:
                continue
            out[str(name)] = (float(bounds[0]), float(bounds[1]))
        if out:
            return out
    return dict(DEFAULT_SELECTIVITY_WINDOWS_MS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze unit selectivity across fixation-category pairs "
            "for multiple PSTH time windows."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_ephys_fixation_psth_config(args.ephys_fixation_psth_cfg)
    settings = FixationPSTHSelectivitySettings(
        cfg_path=args.dataset_cfg,
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        window_stats_filename=cfg.get("selective_window_stats_filename", "window_stats.csv"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        unit_summary_filename=cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        output_pickle_filename=cfg.get("selective_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        windows_ms=_normalize_windows_cfg(cfg.get("selective_windows_ms")),
        alpha=cfg.get("selective_alpha", 0.05),
        test_name=cfg.get("selective_test", "welch_ttest"),
        min_trials_per_condition=cfg.get("selective_min_trials_per_condition", 2),
        use_parallel=cfg.get("selective_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        bin_size_ms_fallback=cfg.get("bin_size_ms", 10.0),
        window_pre_s_fallback=cfg.get("window_pre_s", 1.0),
        window_post_s_fallback=cfg.get("window_post_s", 1.0),
    )

    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True

    result = run_fixation_selectivity_analysis(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_uuids=_as_str_list(args.unit_uuid),
    )
    unit_df = result.get("unit_summary")
    if unit_df is None or unit_df.empty:
        print("[analysis] no unit-level selectivity rows were produced")
        return

    n_units = len(unit_df)
    n_selective = int(unit_df["is_selective_unit"].sum())
    print(f"[analysis] selective units: {n_selective}/{n_units}")
    print("[analysis] wrote fixation selectivity outputs to configured analysis subdir")


if __name__ == "__main__":
    main()

