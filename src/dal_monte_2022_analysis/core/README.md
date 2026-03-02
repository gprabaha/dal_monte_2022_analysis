# core

Pure, reusable logic shared across workflows.

## Scope

`core/` should contain:
- deterministic algorithms and transforms
- domain/statistical primitives
- contracts/datatypes shared across modules

Current subareas:
- `core/behav/`: fixation detection, session cleaning, feature/analysis primitives
- `core/ephys/`: ephys analysis primitives
- `core/signal/`: signal-level operators
- `core/stats/`: reusable hypothesis-testing utilities
- `core/contracts/`: shared contracts

## Non-Goals

`core/` should not contain:
- filesystem traversal or path orchestration
- plotting/export code
- HPC/subprocess orchestration
- CLI argument parsing

Those concerns belong in `runtime/` or `scripts/`.
