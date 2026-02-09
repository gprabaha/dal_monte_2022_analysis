# scripts/plotting

CLI entrypoints for plotting analysis outputs.

Scripts:
- `plot_face_fixation_probability.py`
  Violin plots for within-session and cross-session face fixation probabilities.
- `plot_interactive_face_fixation_probability.py`
  Violin plots for interactive-period fixation probability summaries.
- `plot_out_of_roi_fixation_probability.py`
  Violin plots for within-session and cross-session out-of-ROI probabilities.

Inputs:
- analysis CSV outputs in `analysis_output_root`
- style config in `configs/plotting.yaml`

Outputs:
- PDF figures written under the relevant analysis output subdirectory.
