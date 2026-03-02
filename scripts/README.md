# scripts

CLI entrypoints for running preprocessing, feature extraction, analysis, plotting, and modeling pipelines.

## Structure

- `scripts/behav/`: behavioral pipeline CLIs
- `scripts/ephys/`: ephys pipeline CLIs

Each domain is stage-organized:
- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`
- `bash/` (batch helpers)

## Design Convention

Scripts should remain thin wrappers:
- parse args
- load config
- call package modules in `src/dal_monte_2022_analysis/...`

Domain logic and reusable helpers should not be duplicated here.

## Pipeline References

- Behavioral flow: `scripts/behav/README.md`
- Ephys flow: `scripts/ephys/README.md`
- Full architecture and ordering guide: `docs/repo_design_and_pipelines.md`
