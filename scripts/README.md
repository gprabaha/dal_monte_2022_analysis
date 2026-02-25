# scripts

Thin CLI wrappers over package logic in `src/dal_monte_2022_analysis/`.

Top-level domain folders:
- `scripts/behav/`
- `scripts/ephys/`
- `scripts/combined/`

Each domain folder is split into workflow folders:
- `preprocessing/`
- `features/`
- `analysis/`
- `plotting/`
- `modeling/`
- `bash/`

Design rule: keep orchestration and argument parsing here, keep data logic in `src/`.

Structure rule
- `scripts/` is domain-first and mirrors `src/dal_monte_2022_analysis/{behav,ephys,combined}`.
- Shared infra/config/data helpers stay in `src/dal_monte_2022_analysis/{config,data,utils}`.
