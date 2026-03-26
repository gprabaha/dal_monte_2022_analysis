"""Statistical helpers shared across analyses and plotting."""

from .correction import (
    ALLOWED_PVALUE_CORRECTIONS,
    adjust_pvalues,
    apply_adjusted_pvalues,
    normalize_pvalue_correction,
    reject_nulls,
)
from .hypothesis import (
    paired_ttest_per_lag,
    two_sample_pvalues,
)
from .permutation import (
    permutation_label_statistic_test,
    permutation_mean_difference_test,
)
from .tests import (
    mannwhitneyu_pvalues_per_column,
    one_sample_ttest,
    one_sample_ttest_greater,
    one_sided_pvalue_from_ttest,
    paired_ttest,
    safe_mannwhitneyu,
    safe_one_sample_ttest,
    safe_one_sample_ttest_greater,
    safe_paired_ttest,
    safe_welch_ttest,
    welch_ttest,
)

__all__ = [
    "ALLOWED_PVALUE_CORRECTIONS",
    "normalize_pvalue_correction",
    "adjust_pvalues",
    "apply_adjusted_pvalues",
    "reject_nulls",
    "paired_ttest_per_lag",
    "two_sample_pvalues",
    "permutation_mean_difference_test",
    "permutation_label_statistic_test",
    "safe_welch_ttest",
    "safe_paired_ttest",
    "safe_one_sample_ttest",
    "safe_one_sample_ttest_greater",
    "safe_mannwhitneyu",
    "one_sided_pvalue_from_ttest",
    "welch_ttest",
    "paired_ttest",
    "one_sample_ttest",
    "one_sample_ttest_greater",
    "mannwhitneyu_pvalues_per_column",
]
