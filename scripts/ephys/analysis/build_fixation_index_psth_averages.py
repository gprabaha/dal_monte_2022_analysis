"""Build split and unsplit date-level fixation PSTH averages for preference-index analysis."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.features.fixation_psth import (
    FixationPSTHAverageSettings,
    run_fixation_psth_average_build,
)
from dal_monte_2022_analysis.runtime.io.analysis_index import scan_analysis_date_paths


def _resolve_cli_path(path_text: str) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def _as_str_seq(values):
    if not values:
        return None
    out = [str(v).strip() for v in values if str(v).strip()]
    return out or None


def _ensure_pkl_filename(filename: str) -> str:
    token = str(filename).strip()
    if not token:
        token = "fixations.pkl"
    if not token.endswith(".pkl"):
        token = f"{token}.pkl"
    return token


def _derive_unsplit_filename(split_filename: str) -> str:
    token = _ensure_pkl_filename(split_filename)
    stem = token[:-4]
    if stem.endswith("_split_by_interactive_state"):
        return f"{stem[: -len('_split_by_interactive_state')]}_unsplit_by_interactive_state.pkl"
    return f"{stem}_unsplit_by_interactive_state.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build date-level fixation PSTH averages for preference-index heatmaps "
            "(default 50 ms bins with 25 ms step)."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--target-bin-size-ms", type=float, default=None)
    parser.add_argument("--target-bin-step-ms", type=float, default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--output-subdir-unsplit", default=None)
    parser.add_argument("--output-filename-unsplit", default=None)
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--unsplit-only", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_cfg_path)

    split_output_subdir = cfg.get(
        "selective_index_average_output_subdir_split",
        cfg.get("selective_index_average_output_subdir", "ephys/psth/fixation_psth_index_averages"),
    )
    split_output_filename = _ensure_pkl_filename(
        cfg.get(
            "selective_index_average_output_filename_split",
            cfg.get("selective_index_average_output_filename", "fixations.pkl"),
        ),
    )
    unsplit_output_subdir = cfg.get("selective_index_average_output_subdir_unsplit", split_output_subdir)
    unsplit_output_filename = _ensure_pkl_filename(
        cfg.get(
            "selective_index_average_output_filename_unsplit",
            _derive_unsplit_filename(split_output_filename),
        ),
    )

    settings_common = FixationPSTHAverageSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=split_output_subdir,
        output_filename=split_output_filename,
        split_by_interactive_state=True,
        restrict_interactive_state=cfg.get("selective_index_average_restrict_interactive_state"),
        group_by_session=cfg.get("selective_index_average_group_by_session", False),
        smooth_before_average=cfg.get(
            "selective_index_average_smooth_before_average",
            False,
        ),
        smoothing_sigma_ms=cfg.get(
            "selective_index_average_smoothing_sigma_ms",
            cfg.get("smoothing_sigma_ms", 20.0),
        ),
        target_bin_size_ms=cfg.get("selective_index_average_target_bin_size_ms", 50.0),
        target_bin_step_ms=cfg.get("selective_index_average_target_bin_step_ms", 25.0),
        use_parallel=cfg.get("average_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        categories=cfg.get("selective_index_average_categories", ("face", "object")),
    )

    if args.split_only and args.unsplit_only:
        raise ValueError("Cannot use --split-only and --unsplit-only together.")

    if args.target_bin_size_ms is not None:
        settings_common.target_bin_size_ms = float(args.target_bin_size_ms)
    if args.target_bin_step_ms is not None:
        settings_common.target_bin_step_ms = float(args.target_bin_step_ms)
    if args.no_parallel:
        settings_common.use_parallel = False
    if args.test_single:
        settings_common.test_single = True

    run_split_output = not bool(args.unsplit_only)
    run_unsplit_output = not bool(args.split_only)

    if args.output_subdir:
        split_output_subdir = str(args.output_subdir)
        unsplit_output_subdir = str(args.output_subdir)
    if args.output_subdir_unsplit:
        unsplit_output_subdir = str(args.output_subdir_unsplit)
    if args.output_filename:
        split_output_filename = _ensure_pkl_filename(args.output_filename)
    if args.output_filename_unsplit:
        unsplit_output_filename = _ensure_pkl_filename(args.output_filename_unsplit)

    dates = _as_str_seq(args.date)
    sessions = _as_str_seq(args.session)

    settings_by_label: list[tuple[str, FixationPSTHAverageSettings]] = []
    if run_split_output:
        settings_by_label.append(
            (
                "split",
                replace(
                    settings_common,
                    output_subdir=split_output_subdir,
                    output_filename=split_output_filename,
                    split_by_interactive_state=True,
                ),
            ),
        )
    if run_unsplit_output:
        settings_by_label.append(
            (
                "unsplit",
                replace(
                    settings_common,
                    output_subdir=unsplit_output_subdir,
                    output_filename=unsplit_output_filename,
                    split_by_interactive_state=False,
                ),
            ),
        )
    if not settings_by_label:
        raise ValueError("No index-average output mode selected.")

    for _, mode_settings in settings_by_label:
        run_fixation_psth_average_build(
            mode_settings,
            dates=dates,
            sessions=sessions,
        )

    cfg_dataset = load_config(str(dataset_cfg_path))
    for label, mode_settings in settings_by_label:
        out_rows = scan_analysis_date_paths(
            cfg_dataset,
            mode_settings.output_subdir,
            filename=str(mode_settings.output_filename),
            dates=dates,
        )
        print(
            f"[analysis] built fixation index-input {label} averages: "
            f"subdir={mode_settings.output_subdir}, "
            f"filename={mode_settings.output_filename}, files={len(out_rows)}"
        )


if __name__ == "__main__":
    main()
