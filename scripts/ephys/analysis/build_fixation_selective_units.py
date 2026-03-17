"""Run fixation-pair selective-unit analysis from trial PSTH data."""

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
from dal_monte_2022_analysis.ephys.analysis.fixation_selectivity import (
    DEFAULT_PRIMARY_COMPARISON_GROUP,
    DEFAULT_SELECTIVITY_WINDOWS_MS,
    FixationPSTHSelectivitySettings,
    run_fixation_selectivity_analysis,
)


def _as_str_list(values):
    if not values:
        return None
    return [str(v) for v in values if str(v).strip()]


def _scan_trial_files(dataset_cfg_path: str, modality: str, filename: str) -> tuple[Path, int, int, int]:
    dataset_cfg = load_config(dataset_cfg_path)
    root = Path(dataset_cfg["processed_data_root"])
    base_glob = root / "date=*" / "session=*" / str(modality) / "*.pkl"
    all_rows = list(root.glob(str(base_glob.relative_to(root))))
    matched_rows = [path for path in all_rows if path.name == str(filename)]
    shared_rows = [path for path in all_rows if path.name == "shared.pkl"]
    return root, len(all_rows), len(matched_rows), len(shared_rows)


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


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze unit selectivity across fixation-category pairs "
            "for multiple PSTH time windows."
        ),
    )
    parser.add_argument("--dataset-cfg", default=str(_DEFAULT_DATASET_CFG))
    parser.add_argument("--ephys-fixation-psth-cfg", default=str(_DEFAULT_EPHYS_FIX_PSTH_CFG))
    parser.add_argument("--date", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--unit-uuid", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_psth_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_psth_cfg_path)
    settings = FixationPSTHSelectivitySettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("selective_trial_input_modality", cfg.get("trial_output_modality", "psth")),
        trial_input_filename=cfg.get("selective_trial_input_filename", "fixations_psth_10ms.pkl"),
        output_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        window_stats_filename=cfg.get("selective_window_stats_filename", "window_stats.csv"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        unit_summary_filename=cfg.get("selective_unit_summary_filename", "unit_selectivity.csv"),
        condition_summary_filename=cfg.get("selective_condition_summary_filename", "condition_window_means.csv"),
        output_pickle_filename=cfg.get("selective_output_pickle_filename", "results.pkl"),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        windows_ms=_normalize_windows_cfg(cfg.get("selective_windows_ms")),
        significance_windows=tuple(
            cfg.get("selective_significance_windows", ("pre_fix", "peri_fix", "post_fix"))
        ),
        comparison_groups=cfg.get("selective_comparison_groups"),
        primary_comparison_group=cfg.get(
            "selective_primary_comparison_group",
            DEFAULT_PRIMARY_COMPARISON_GROUP,
        ),
        smooth_before_window_average=cfg.get(
            "selective_smooth_before_window_average",
            cfg.get("smooth_before_average", True),
        ),
        smoothing_sigma_ms=cfg.get(
            "selective_smoothing_sigma_ms",
            cfg.get("smoothing_sigma_ms", 20.0),
        ),
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

    root, n_all, n_match, n_shared = _scan_trial_files(
        str(dataset_cfg_path),
        settings.trial_input_modality,
        settings.trial_input_filename,
    )
    if n_match == 0 and settings.trial_input_filename != "shared.pkl" and n_shared > 0:
        print(
            "[analysis] no files found for "
            f"{settings.trial_input_modality}/{settings.trial_input_filename}; "
            "falling back to shared.pkl."
        )
        settings.trial_input_filename = "shared.pkl"
        _, n_all, n_match, n_shared = _scan_trial_files(
            str(dataset_cfg_path),
            settings.trial_input_modality,
            settings.trial_input_filename,
        )

    print(
        "[analysis] config paths: "
        f"dataset={dataset_cfg_path}, "
        f"ephys_fixation_psth={ephys_fix_psth_cfg_path}"
    )
    print(
        "[analysis] trial file scan: "
        f"root={root}, modality={settings.trial_input_modality}, "
        f"filename={settings.trial_input_filename}, matched={n_match}, "
        f"all_pkls_in_modality={n_all}, shared_pkls={n_shared}"
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
        print(
            "[analysis] if matched=0 above, check dataset.yaml processed_data_root "
            "or run with --dataset-cfg pointing to the config used for PSTH trial generation."
        )
        return

    n_units = len(unit_df)
    n_selective = int(unit_df["is_selective_unit"].sum())
    condition_df = result.get("condition_summary")
    n_condition_rows = 0 if condition_df is None else int(len(condition_df))
    comparison_results = result.get("comparison_results", {})
    print(
        "[analysis] comparison groups evaluated: "
        f"{sorted(str(key) for key in comparison_results.keys())}"
    )
    print(f"[analysis] selective units: {n_selective}/{n_units}")
    print(f"[analysis] three-way condition summary rows: {n_condition_rows}")
    print("[analysis] wrote fixation selectivity outputs to configured analysis subdir")


if __name__ == "__main__":
    main()
