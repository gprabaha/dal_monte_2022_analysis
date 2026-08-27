"""Thin entry point for behavior overview tables and figures.

Run from the repository root with:
    conda run -n gaze_processing python notebooks/behavior/build_behavior_overview.py
"""

from dal_monte_2022_analysis.behav.analysis.behavior_overview import build_behavior_overview


if __name__ == "__main__":
    build_behavior_overview()
