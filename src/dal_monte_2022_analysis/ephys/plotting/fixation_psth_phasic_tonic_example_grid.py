"""Plot a 2x4 fixation PSTH example grid for phasic and tonic units."""

from __future__ import annotations

from typing import Sequence

from dal_monte_2022_analysis.ephys.plotting.fixation_psth_example_grid import (
    DEFAULT_EXAMPLE_GRID_REGIONS,
    FixationPSTHExampleGridPlotSettings,
    FixationPSTHExampleUnitSpec,
    parse_example_grid_unit_specs,
    plot_fixation_psth_example_grid,
)


DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS = DEFAULT_EXAMPLE_GRID_REGIONS
DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES = ("phasic", "tonic")
DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_LABELS = {
    "phasic": "Phasic",
    "tonic": "Tonic",
}

_ROW_STYLE_ALIASES = {
    "phasic": "phasic",
    "classic_phasic": "phasic",
    "classic": "phasic",
    "tonic": "tonic",
    "sustained": "tonic",
}


def normalize_example_response_style(style: object) -> str:
    """Normalize a phasic/tonic example-grid row key."""
    token = str(style).strip().lower().replace("-", "_").replace(" ", "_")
    resolved = _ROW_STYLE_ALIASES.get(token)
    if resolved is None:
        supported = ", ".join(DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES)
        raise ValueError(
            f"Unsupported response style '{style}'. Expected one of: {supported}.",
        )
    return resolved


def parse_phasic_tonic_example_grid_unit_specs(
    cfg: dict,
    *,
    regions: Sequence[str] = DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_REGIONS,
    row_styles: Sequence[str] = DEFAULT_PHASIC_TONIC_EXAMPLE_GRID_ROW_STYLES,
    cfg_key: str = "phasic_tonic_example_grid_units",
) -> list[FixationPSTHExampleUnitSpec]:
    """Parse manual phasic/tonic example-unit selections from config."""
    return parse_example_grid_unit_specs(
        cfg,
        regions=regions,
        row_preferences=row_styles,
        cfg_key=cfg_key,
        row_key_normalizer=normalize_example_response_style,
    )


def plot_fixation_psth_phasic_tonic_example_grid(
    settings: FixationPSTHExampleGridPlotSettings,
    *,
    unit_specs: Sequence[FixationPSTHExampleUnitSpec],
    sessions: Sequence[str] | None = None,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Render one 2x4 fixation PSTH example grid with rows=phasic/tonic."""
    return plot_fixation_psth_example_grid(
        settings,
        unit_specs=unit_specs,
        sessions=sessions,
        allow_missing=allow_missing,
        row_key_normalizer=normalize_example_response_style,
    )
