"""Plot fixation population PCA pairwise-geometry violins with significance bars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.analysis.fixation_population_pca import (
    resolve_population_pca_variant_config,
)
from dal_monte_2022_analysis.ephys.plotting.fixation_population_pca import (
    FixationPopulationPCAPlotSettings,
    plot_fixation_population_pca_pairwise_geometry_violins,
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


def _normalize_str_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        seq = raw
    else:
        seq = [raw]
    out: list[str] = []
    for item in seq:
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot four violin summaries for fixation population PCA pairwise geometry "
            "(within/cross region x Euclidean distance/angle)."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--analysis-variant", default=None)
    parser.add_argument("--region", action="append", default=None)
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    variant_cfg = resolve_population_pca_variant_config(
        cfg,
        analysis_variant=args.analysis_variant,
    )
    pca_subdir = variant_cfg.get(
        "output_subdir",
        cfg.get("population_pca_output_subdir", "ephys/psth/fixation_population_pca"),
    )
    settings = FixationPopulationPCAPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        input_subdir=str(variant_cfg.get("plot_input_subdir", cfg.get("population_pca_plot_input_subdir", pca_subdir))),
        input_filename=cfg.get(
            "population_pca_plot_input_filename",
            cfg.get("population_pca_output_pickle_filename", "results.pkl"),
        ),
        output_subdir=str(
            variant_cfg.get(
                "plot_output_subdir",
                cfg.get(
                    "population_pca_plot_output_subdir",
                    f"{str(pca_subdir).rstrip('/')}/plots",
                ),
            )
        ),
        output_extension=cfg.get("population_pca_plot_output_extension", "pdf"),
        output_dpi=cfg.get("population_pca_plot_output_dpi", 300),
        conditions=tuple(_normalize_str_list(variant_cfg.get("conditions"))),
        condition_labels=_normalize_condition_colors(variant_cfg.get("plot_condition_labels", {})),
        condition_colors=_normalize_condition_colors(
            variant_cfg.get(
                "plot_condition_colors",
                cfg.get("population_pca_plot_condition_colors", cfg.get("plot_condition_colors", {})),
            ),
        ),
        pairwise_violin_letter_width_in=float(
            cfg.get("population_pca_plot_pairwise_violin_letter_width_in", 8.5)
        ),
        pairwise_violin_letter_height_frac=float(
            cfg.get("population_pca_plot_pairwise_violin_letter_height_frac", 0.28)
        ),
    )

    outputs = plot_fixation_population_pca_pairwise_geometry_violins(
        settings,
        regions=args.region,
        output_filename=str(
            variant_cfg.get(
                "plot_pairwise_geometry_output_filename",
                cfg.get(
                    "population_pca_plot_pairwise_geometry_output_filename",
                    "population_pca_pairwise_geometry_violin",
                ),
            )
        ),
    )
    if not outputs:
        print("[plot] no pairwise-geometry violin figures generated")
        return
    for row in outputs:
        print(
            "[plot] pairwise-geometry figure: "
            f"{row.get('output_path')} "
            f"(metric={row.get('metric_name')}, kind={row.get('kind')})"
        )


if __name__ == "__main__":
    main()
