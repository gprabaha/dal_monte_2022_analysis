"""Compatibility wrapper for monkey-role leader/follower plotting module."""

from dal_monte_2022_analysis.behav.plotting.fix_cross_correlation_leader_follower_monkey_role import (
    LeaderFollowerMonkeyRoleFixationCountPlotSettings,
    LeaderFollowerMonkeyRoleFixationDurationPlotSettings,
    LeaderFollowerMonkeyRolePupilPlotSettings,
    LeaderFollowerPupilGlobalOverlayPlotSettings,
    plot_leader_follower_monkey_role_fixation_count_violin,
    plot_leader_follower_monkey_role_fixation_duration_violin,
    plot_leader_follower_monkey_role_pupil_violin,
    plot_leader_follower_pupil_global_overlay_violin,
)

__all__ = [
    "LeaderFollowerMonkeyRolePupilPlotSettings",
    "LeaderFollowerPupilGlobalOverlayPlotSettings",
    "LeaderFollowerMonkeyRoleFixationDurationPlotSettings",
    "LeaderFollowerMonkeyRoleFixationCountPlotSettings",
    "plot_leader_follower_pupil_global_overlay_violin",
    "plot_leader_follower_monkey_role_pupil_violin",
    "plot_leader_follower_monkey_role_fixation_duration_violin",
    "plot_leader_follower_monkey_role_fixation_count_violin",
]
