"""Plot fixation population PCA trajectories in top-3 PC space."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_population_pca import (
    FixationPopulationPCAPlotSettings,
    plot_fixation_population_pca_trajectories,
)


def _normalize_condition_colors(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top-3 PC trajectories for fixation population PCA outputs.",
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    pca_subdir = cfg.get("population_pca_output_subdir", "ephys/psth/fixation_population_pca")
    settings = FixationPopulationPCAPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=cfg.get("population_pca_plot_input_subdir", pca_subdir),
        input_filename=cfg.get(
            "population_pca_plot_input_filename",
            cfg.get("population_pca_output_pickle_filename", "results.pkl"),
        ),
        output_subdir=cfg.get(
            "population_pca_plot_output_subdir",
            f"{str(pca_subdir).rstrip('/')}/plots",
        ),
        output_extension=cfg.get("population_pca_plot_output_extension", "pdf"),
        output_dpi=cfg.get("population_pca_plot_output_dpi", 300),
        trajectory_n_pcs=int(cfg.get("population_pca_plot_trajectory_n_pcs", 3)),
        trajectory_n_columns=int(cfg.get("population_pca_plot_trajectory_n_columns", 4)),
        condition_colors=_normalize_condition_colors(
            cfg.get("population_pca_plot_condition_colors", cfg.get("plot_condition_colors", {})),
        ),
    )

    out = plot_fixation_population_pca_trajectories(
        settings,
        regions=args.region,
        output_filename=cfg.get(
            "population_pca_plot_trajectory_output_filename",
            "population_pca_pc_trajectories",
        ),
    )
    if not out:
        print("[plot] no trajectory figure generated")
        return
    print(f"[plot] trajectory figure: {out.get('output_path')}")
    print(f"[plot] regions: {out.get('regions')}")


if __name__ == "__main__":
    main()

