"""Compatibility shim for pickle migration helpers.

Canonical import path:
`dal_monte_2022_analysis.data.migrations.pickle_modules`.
"""

from dal_monte_2022_analysis.data.migrations.pickle_modules import (  # noqa: F401
    PickleMigrationRecord,
    PickleMigrationSummary,
    migrate_legacy_pickle_modules,
)

__all__ = [
    "PickleMigrationRecord",
    "PickleMigrationSummary",
    "migrate_legacy_pickle_modules",
]
