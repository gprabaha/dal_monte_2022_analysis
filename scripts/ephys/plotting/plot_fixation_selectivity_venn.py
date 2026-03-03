"""Plot region-level Venn diagrams for fixation selectivity pairs."""

import argparse

from dal_monte_2022_analysis.config.load import load_config
from dal_monte_2022_analysis.ephys.plotting.fixation_selectivity_venn import (
    FixationSelectivityVennPlotSettings,
    build_fixation_selectivity_venn_summaries,
)


def _print_region_summary(summary: dict) -> None:
    region = str(summary["region"])
    total = int(summary["total_units"])
    any_sel = int(summary["any_selective"])
    set_counts = summary["set_counts"]
    seg = summary["segment_counts"]
    out_path = summary.get("output_path")

    def pct(n: int) -> float:
        return 0.0 if total <= 0 else (100.0 * float(n) / float(total))

    print(f"\n[region] {region}")
    print(f"  total_units: {total}")
    print(f"  any_selective: {any_sel} ({pct(any_sel):.1f}%)")
    print("  set_counts:")
    print(
        "    int_face_vs_nonint_face: "
        f"{set_counts['face_int_vs_nonint']} ({pct(set_counts['face_int_vs_nonint']):.1f}%)"
    )
    print(
        "    int_face_vs_object: "
        f"{set_counts['face_int_vs_obj']} ({pct(set_counts['face_int_vs_obj']):.1f}%)"
    )
    print(
        "    nonint_face_vs_object: "
        f"{set_counts['face_nonint_vs_obj']} ({pct(set_counts['face_nonint_vs_obj']):.1f}%)"
    )
    print("  venn_segments:")
    print(f"    A_only: {seg['a_only']} ({pct(seg['a_only']):.1f}%)")
    print(f"    B_only: {seg['b_only']} ({pct(seg['b_only']):.1f}%)")
    print(f"    C_only: {seg['c_only']} ({pct(seg['c_only']):.1f}%)")
    print(f"    AB_only: {seg['ab_only']} ({pct(seg['ab_only']):.1f}%)")
    print(f"    AC_only: {seg['ac_only']} ({pct(seg['ac_only']):.1f}%)")
    print(f"    BC_only: {seg['bc_only']} ({pct(seg['bc_only']):.1f}%)")
    print(f"    ABC: {seg['abc']} ({pct(seg['abc']):.1f}%)")
    if out_path is not None:
        print(f"  figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot region-level Venn diagrams for fixation selectivity pairs and "
            "print region counts/percentages to terminal."
        ),
    )
    parser.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    parser.add_argument("--plotting-cfg", default="configs/plotting.yaml")
    parser.add_argument("--ephys-fixation-psth-cfg", default="configs/ephys_fixation_psth.yaml")
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--test-single", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.ephys_fixation_psth_cfg)
    settings = FixationSelectivityVennPlotSettings(
        cfg_path=args.dataset_cfg,
        plotting_cfg_path=args.plotting_cfg,
        selectivity_input_subdir=cfg.get("selective_output_subdir", "ephys/psth/fixation_psth_selectivity"),
        pair_summary_filename=cfg.get("selective_pair_summary_filename", "pair_selectivity.csv"),
        output_subdir=cfg.get("selective_venn_output_subdir", "ephys/psth/fixation_psth_selectivity_venn"),
        output_extension=cfg.get("selective_venn_output_extension", "pdf"),
        output_dpi=cfg.get("selective_venn_output_dpi", 220),
        use_parallel=cfg.get("selective_venn_use_parallel", True),
        max_procs=cfg.get("max_procs", 16),
        test_single=cfg.get("test_single", False),
        min_units_per_region=cfg.get("selective_venn_min_units_per_region", 1),
        combine_regions_into_single_figure=cfg.get("selective_venn_combine_regions_into_single_figure", True),
        region_order=cfg.get("selective_venn_region_order", ["BLA", "ACCg", "dmPFC", "OFC"]),
        combined_output_filename=cfg.get("selective_venn_combined_output_filename", "regions_1x4"),
        combined_figure_width_in=cfg.get("selective_venn_combined_figure_width_in", 8.5),
        combined_figure_height_in=cfg.get("selective_venn_combined_figure_height_in", 2.2),
        combined_left_margin=cfg.get("selective_venn_combined_left_margin", 0.01),
        combined_right_margin=cfg.get("selective_venn_combined_right_margin", 0.995),
        combined_top_margin=cfg.get("selective_venn_combined_top_margin", 0.79),
        combined_bottom_margin=cfg.get("selective_venn_combined_bottom_margin", 0.06),
        combined_wspace=cfg.get("selective_venn_combined_wspace", 0.15),
        combined_show_pair_key=cfg.get("selective_venn_combined_show_pair_key", True),
    )

    if args.no_parallel:
        settings.use_parallel = False
    if args.test_single:
        settings.test_single = True

    outputs = build_fixation_selectivity_venn_summaries(
        settings,
        regions=args.region,
    )
    if not outputs:
        print("[plot] no region Venn plots were produced")
        return

    for summary in outputs:
        _print_region_summary(summary)
    unique_paths = sorted({str(row.get("output_path")) for row in outputs if row.get("output_path") is not None})
    print(
        f"\n[plot] wrote {len(unique_paths)} Venn figure(s) "
        f"covering {len(outputs)} region summaries"
    )


if __name__ == "__main__":
    main()
