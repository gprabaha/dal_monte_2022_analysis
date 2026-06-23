"""Minimal Elman mRNN model for fixation PSTH targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FixationMRNNModelSpec:
    """Everything needed to construct the model."""

    region_order: tuple[str, ...]
    hidden_units_by_region: dict[str, int]
    output_dims_by_region: dict[str, int]
    input_dim: int = 3
    input_region_name: str = "input"
    activation: str = "softplus"
    spectral_radius: float | None = 1.3
    rec_constrained: bool = False
    inp_constrained: bool = False
    recurrent_connectivity: str = "full"
    batch_first: bool = True
    inp_noise: float = 0.0
    act_noise: float = 0.0
    device: str = "cpu"


class FixationMRNNModel(nn.Module):
    """Elman mRNN plus one linear readout per region."""

    def __init__(self, spec: FixationMRNNModelSpec):
        super().__init__()
        self.spec = spec
        self.region_order = tuple(spec.region_order)
        try:
            from mrnntorch import ElmanmRNN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("The lowercase 'mrnntorch' package is required.") from exc

        self.mrnn = ElmanmRNN(
            activation=spec.activation,
            noise_level_act=float(spec.act_noise),
            noise_level_inp=float(spec.inp_noise),
            rec_constrained=bool(spec.rec_constrained),
            inp_constrained=bool(spec.inp_constrained),
            batch_first=bool(spec.batch_first),
            spectral_radius=spec.spectral_radius,
            config_finalize=False,
            device=str(spec.device),
        )
        for region in self.region_order:
            self.mrnn.add_recurrent_region(
                region,
                int(spec.hidden_units_by_region[region]),
                sign="pos",
                device=str(spec.device),
                learnable_bias=True,
            )
        self.mrnn.add_input_region(
            spec.input_region_name,
            int(spec.input_dim),
            sign="pos",
            device=str(spec.device),
        )
        for source in self.region_order:
            for target in self.region_order:
                if _region_pair_connected(source, target, spec.recurrent_connectivity):
                    self.mrnn.add_recurrent_connection(source, target)
        for target in self.region_order:
            self.mrnn.add_input_connection(spec.input_region_name, target)
        self.mrnn.finalize_connectivity()
        if normalize_recurrent_connectivity(spec.recurrent_connectivity) == "cross_region_with_self_diagonal":
            self._enable_self_diagonal_recurrent_mask()

        self.output_heads = nn.ModuleDict(
            {
                region: nn.Linear(
                    int(spec.hidden_units_by_region[region]),
                    int(spec.output_dims_by_region[region]),
                )
                for region in self.region_order
            }
        )

    @property
    def total_num_units(self) -> int:
        return int(self.mrnn.total_num_units)

    def hidden_region_slices(self) -> dict[str, slice]:
        """Return hidden-state slices for each recurrent region."""
        slices = {}
        for region in self.region_order:
            start, stop = self.mrnn.get_region_indices(region)
            slices[region] = slice(int(start), int(stop))
        return slices

    def _enable_self_diagonal_recurrent_mask(self) -> None:
        """Enable only matched-unit self dynamics inside each region block."""
        with torch.no_grad():
            for region in self.region_order:
                region_slice = self.hidden_region_slices()[region]
                block = self.mrnn.W_rec_mask[region_slice, region_slice]
                block.zero_()
                diagonal_count = min(block.shape)
                idx = torch.arange(diagonal_count, device=block.device)
                block[idx, idx] = 1.0
                sign_block = self.mrnn.W_rec_sign_matrix[region_slice, region_slice]
                sign_block[idx, idx] = 1.0

    def output_region_slices(self) -> dict[str, slice]:
        """Return concatenated-output slices for each readout region."""
        slices = {}
        start = 0
        for region in self.region_order:
            stop = start + int(self.spec.output_dims_by_region[region])
            slices[region] = slice(start, stop)
            start = stop
        return slices

    def readout_weight_matrix(self) -> torch.Tensor:
        """Return the explicit block-local hidden-to-output readout matrix.

        The forward path uses one linear head per region. This helper exposes
        the equivalent concatenated matrix, with zeros outside each
        region's own hidden-to-output block.
        """
        first_weight = next(iter(self.output_heads.values())).weight
        weight = torch.zeros(
            sum(int(self.spec.output_dims_by_region[region]) for region in self.region_order),
            self.total_num_units,
            dtype=first_weight.dtype,
            device=first_weight.device,
        )
        hidden_slices = self.hidden_region_slices()
        output_slices = self.output_region_slices()
        for region in self.region_order:
            weight[output_slices[region], hidden_slices[region]] = self.output_heads[region].weight
        return weight

    def readout_bias_vector(self) -> torch.Tensor:
        """Return the concatenated output bias ordered by region."""
        return torch.cat([self.output_heads[region].bias for region in self.region_order], dim=0)

    def forward(
        self,
        inp: torch.Tensor,
        h0: torch.Tensor,
        *,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
    ) -> dict[str, object]:
        """Run the Elman mRNN and read out every region."""
        h_seq = self.mrnn(inp, h0, stim_input=stim_input, noise=bool(noise))
        output_by_region = {}
        for region in self.region_order:
            region_h = self.mrnn.get_region_activity(h_seq, region)
            output_by_region[region] = self.output_heads[region](region_h)
        output = torch.cat([output_by_region[region] for region in self.region_order], dim=-1)
        return {
            "output": output,
            "output_by_region": output_by_region,
            "h_seq": h_seq,
        }


def build_model_spec(
    *,
    region_order: tuple[str, ...],
    output_dims_by_region: dict[str, int],
    hidden_units: int | dict[str, int],
    device: str,
    **kwargs,
) -> FixationMRNNModelSpec:
    """Build a spec, expanding scalar hidden units across regions."""
    if isinstance(hidden_units, dict):
        hidden_by_region = {region: int(hidden_units[region]) for region in region_order}
    else:
        hidden_by_region = {region: int(hidden_units) for region in region_order}
    return FixationMRNNModelSpec(
        region_order=tuple(region_order),
        hidden_units_by_region=hidden_by_region,
        output_dims_by_region={region: int(output_dims_by_region[region]) for region in region_order},
        device=str(device),
        **kwargs,
    )


RECURRENT_CONNECTIVITY_ALIASES = {
    "full": "full",
    "all": "full",
    "all_region": "full",
    "all_region_to_region": "full",
    "all_regions": "full",
    "within": "within_region",
    "within_region": "within_region",
    "internal": "within_region",
    "internal_only": "within_region",
    "cross": "cross_region_with_self_diagonal",
    "cross_region": "cross_region_with_self_diagonal",
    "cross_region_only": "cross_region_with_self_diagonal",
    "cross_region_with_self_diagonal": "cross_region_with_self_diagonal",
    "cross_plus_diagonal": "cross_region_with_self_diagonal",
}


def normalize_recurrent_connectivity(connectivity: str) -> str:
    """Normalize recurrent connectivity aliases."""
    token = str(connectivity).strip().lower().replace("-", "_")
    try:
        return RECURRENT_CONNECTIVITY_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "recurrent_connectivity must be one of: "
            "'full', 'within_region', or 'cross_region_with_self_diagonal'."
        ) from exc


def _region_pair_connected(source: str, target: str, connectivity: str) -> bool:
    mode = normalize_recurrent_connectivity(connectivity)
    if mode == "full":
        return True
    if mode == "within_region":
        return source == target
    if mode == "cross_region_with_self_diagonal":
        return source != target
    raise ValueError(f"Unsupported recurrent connectivity: {connectivity!r}")


__all__ = [
    "FixationMRNNModel",
    "FixationMRNNModelSpec",
    "build_model_spec",
    "normalize_recurrent_connectivity",
]
