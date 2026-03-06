"""Build ROI-vs-period factorial fixation analysis outputs."""

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
from dal_monte_2022_analysis.ephys.analysis.fixation_roi_vs_period_factorial import (
    DEFAULT_FACTORIAL_WINDOWS_MS,
    DEFAULT_SIGNIFICANCE_WINDOWS,
    FixationROIVsPeriodFactorialSettings,
    print_fixation_roi_vs_period_factorial_summary,
    run_fixation_roi_vs_period_factorial_analysis,
)


def _as_str_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def _normalize_windows_cfg(raw):
    if raw is None:
        return dict(DEFAULT_FACTORIAL_WINDOWS_MS)
    if isinstance(raw, dict):
        out = {}
        for name, bounds in raw.items():
            if bounds is None or len(bounds) != 2:
                continue
            out[str(name)] = (float(bounds[0]), float(bounds[1]))
        if out:
            return out
    return dict(DEFAULT_FACTORIAL_WINDOWS_MS)


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-unit ROI-vs-period factorial models from trial fixation PSTHs, "
            "then run region-level significance and axis-comparison summaries."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_DEFAULT_DATASET_CFG))
    parser.add_argument("--ephys-fixation-psth-cfg", default=str(_DEFAULT_EPHYS_FIX_PSTH_CFG))
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--window", action="append", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    settings = FixationROIVsPeriodFactorialSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get(
            "roi_vs_period_output_subdir",
            "ephys/psth/fixation_roi_vs_period_factorial",
        ),
        unit_term_filename=cfg.get("roi_vs_period_unit_term_filename", "unit_glm_terms.csv"),
        unit_axis_filename=cfg.get("roi_vs_period_unit_axis_filename", "unit_axis_values.csv"),
        unit_window_summary_filename=cfg.get(
            "roi_vs_period_unit_window_summary_filename",
            "unit_window_condition_means.csv",
        ),
        region_fraction_filename=cfg.get(
            "roi_vs_period_region_fraction_filename",
            "region_significant_fractions.csv",
        ),
        region_fraction_pairwise_filename=cfg.get(
            "roi_vs_period_region_fraction_pairwise_filename",
            "region_significant_fraction_pairwise.csv",
        ),
        region_fraction_within_region_filename=cfg.get(
            "roi_vs_period_region_fraction_within_region_filename",
            "region_significant_fraction_within_region.csv",
        ),
        region_axis_summary_filename=cfg.get(
            "roi_vs_period_region_axis_summary_filename",
            "region_axis_summary.csv",
        ),
        region_axis_pairwise_filename=cfg.get(
            "roi_vs_period_region_axis_pairwise_filename",
            "region_axis_pairwise.csv",
        ),
        region_axis_within_region_filename=cfg.get(
            "roi_vs_period_region_axis_within_region_filename",
            "region_axis_within_region.csv",
        ),
        region_axis_friedman_filename=cfg.get(
            "roi_vs_period_region_axis_friedman_filename",
            "region_axis_friedman.csv",
        ),
        output_pickle_filename=cfg.get("roi_vs_period_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        windows_ms=_normalize_windows_cfg(cfg.get("roi_vs_period_windows_ms", cfg.get("selective_windows_ms"))),
        significance_windows=tuple(
            cfg.get("roi_vs_period_significance_windows", tuple(DEFAULT_SIGNIFICANCE_WINDOWS))
        ),
        smooth_before_window_average=cfg.get(
            "roi_vs_period_smooth_before_window_average",
            cfg.get("selective_smooth_before_window_average", cfg.get("smooth_before_average", True)),
        ),
        smoothing_sigma_ms=cfg.get(
            "roi_vs_period_smoothing_sigma_ms",
            cfg.get("selective_smoothing_sigma_ms", cfg.get("smoothing_sigma_ms", 20.0)),
        ),
        min_trials_per_cell=cfg.get("roi_vs_period_min_trials_per_cell", 2),
        min_units_per_region=cfg.get("roi_vs_period_min_units_per_region", 5),
        alpha=cfg.get("roi_vs_period_alpha", 0.05),
        pvalue_correction=str(cfg.get("roi_vs_period_pvalue_correction") or "fdr_bh"),
        unit_significance_mode=cfg.get("roi_vs_period_unit_significance_mode", "within_unit_corrected"),
        use_parallel=cfg.get("roi_vs_period_use_parallel", True),
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

    print(
        "[analysis] roi-vs-period factorial request: "
        f"date={args.date or 'all'}, session={args.session or 'all'}, "
        f"regions={args.region or 'all'}, windows={args.window or 'all'}"
    )
    print(
        "[analysis] roi-vs-period factorial settings: "
        f"trial_source={settings.trial_input_modality}/{settings.trial_input_filename}, "
        f"smooth_before_window_average={bool(settings.smooth_before_window_average)}, "
        f"smoothing_sigma_ms={float(settings.smoothing_sigma_ms):.3f}, "
        f"min_trials_per_cell={int(settings.min_trials_per_cell)}, "
        f"pvalue_correction={settings.pvalue_correction}, "
        f"unit_significance_mode={settings.unit_significance_mode}"
    )

    result = run_fixation_roi_vs_period_factorial_analysis(
        settings,
        dates=[args.date] if args.date else None,
        sessions=[args.session] if args.session else None,
        unit_uuids=_as_str_list(args.unit_uuid),
        regions=_as_str_list(args.region),
        windows=_as_str_list(args.window),
    )
    print_fixation_roi_vs_period_factorial_summary(result)

    print("[analysis] wrote roi-vs-period factorial outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
