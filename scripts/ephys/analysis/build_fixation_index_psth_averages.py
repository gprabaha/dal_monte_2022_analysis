"""Deprecated wrapper for index-input average builder.

Use:
  scripts/ephys/analysis/build_fixation_preference_index_wide_binned_firing_rate_averages.py
"""

from pathlib import Path
import sys


def main() -> None:
    print(
        "[analysis] DEPRECATED script name: build_fixation_index_psth_averages.py\n"
        "[analysis] Please use: build_fixation_preference_index_wide_binned_firing_rate_averages.py"
    )
    from build_fixation_preference_index_wide_binned_firing_rate_averages import main as _main

    _main()


if __name__ == "__main__":
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    main()
