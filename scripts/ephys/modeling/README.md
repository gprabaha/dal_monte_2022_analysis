# scripts/ephys/modeling

Thin CLIs for fixation mRNN modeling.

- `prepare_fixation_mrnn_experiment.py`: create an indexed `run_plan.csv`.
- `inspect_fixation_mrnn_targets.py`: print target snippets, PCA dimensions, and shuffle diagnostics.
- `train_fixation_mrnn.py`: train one scratch run or one indexed experiment run.
- `index_fixation_mrnn_experiment.py`: rebuild `index.csv` from run outputs.
- `analyze_fixation_mrnn_run.py`: run lightweight current/eigenvalue/flow-field checks.
