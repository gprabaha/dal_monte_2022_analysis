"""Build the trial-count-matched CV control for fixation PSTHs."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_cv_trial_matched import (
    FixationCVTrialMatchedSettings,
    run_fixation_cv_trial_matched_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-condition CV of the mean fixation PSTH from "
            "trial-count-matched subsamples, removing the estimation-noise "
            "advantage that the more numerous interactive-face trials enjoy."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--n-draws", type=int, default=None)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    dataset_cfg_path = Path(args.dataset_cfg).expanduser().resolve()
    cfg = load_config(Path(args.ephys_fixation_psth_cfg).expanduser().resolve())

    settings = FixationCVTrialMatchedSettings(
        cfg_path=str(dataset_cfg_path),
        trial_input_modality=cfg.get("trial_output_modality", "psth"),
        trial_input_filename=cfg.get(
            "variability_trial_input_filename",
            cfg.get("plot_trial_input_filename", "fixations_psth_10ms.pkl"),
        ),
        interactive_label=cfg.get("interactive_high_label", "interactive"),
        face_label=cfg.get("face_label", "face"),
        object_label=cfg.get("object_label", "object"),
        smoothing_sigma_ms=float(cfg.get("average_smoothing_sigma_ms", 20.0)),
        bin_size_ms_fallback=float(cfg.get("bin_size_ms", 10.0)),
        pvalue_correction=cfg.get("selective_pvalue_correction", "fdr_bh"),
        alpha=float(cfg.get("selective_alpha", 0.05)),
        use_parallel=not args.no_parallel,
        max_procs=int(cfg.get("max_procs", 16)),
    )
    if args.n_draws is not None:
        settings.n_draws = int(args.n_draws)

    result = run_fixation_cv_trial_matched_analysis(settings, dates=args.date)
    unit_df = result["unit_cv"]
    stats_df = result["within_region_stats"]
    inflation = result["inflation"]
    print(f"[analysis] units with all conditions: {len(unit_df)}")
    print(f"[analysis] within-region tests:       {len(stats_df)}")
    print(f"[analysis] inflation-curve rows:      {len(inflation)}")
    if len(unit_df):
        print(f"[analysis] median matched trial count: {unit_df['matched_n_trials'].median():.0f}")


if __name__ == "__main__":
    main()
