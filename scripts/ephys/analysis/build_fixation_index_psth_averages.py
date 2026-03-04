"""Build date-level fixation PSTH averages for preference-index heatmap analysis."""

import argparse
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
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = _resolve_cli_path(args.dataset_cfg)
    ephys_fix_cfg_path = _resolve_cli_path(args.ephys_fixation_psth_cfg)
    cfg = load_config(ephys_fix_cfg_path)

    settings = FixationPSTHAverageSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get("trial_output_filename", "fixations.pkl"),
        output_subdir=cfg.get("selective_index_average_output_subdir", "ephys/psth/fixation_psth_index_averages"),
        output_filename=cfg.get("selective_index_average_output_filename", "fixations.pkl"),
        split_by_interactive_state=cfg.get("selective_index_average_split_by_interactive_state", True),
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

    if args.target_bin_size_ms is not None:
        settings.target_bin_size_ms = float(args.target_bin_size_ms)
    if args.target_bin_step_ms is not None:
        settings.target_bin_step_ms = float(args.target_bin_step_ms)
    if args.output_subdir:
        settings.output_subdir = str(args.output_subdir)
    if args.output_filename:
        settings.output_filename = str(args.output_filename)
    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True

    dates = _as_str_seq(args.date)
    sessions = _as_str_seq(args.session)

    run_fixation_psth_average_build(
        settings,
        dates=dates,
        sessions=sessions,
    )

    cfg_dataset = load_config(str(dataset_cfg_path))
    out_rows = scan_analysis_date_paths(
        cfg_dataset,
        settings.output_subdir,
        filename=str(settings.output_filename),
        dates=dates,
    )
    print(
        "[analysis] built fixation index-input averages: "
        f"subdir={settings.output_subdir}, filename={settings.output_filename}, files={len(out_rows)}"
    )


if __name__ == "__main__":
    main()
