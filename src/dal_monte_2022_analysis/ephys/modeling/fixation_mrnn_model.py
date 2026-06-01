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
                self.mrnn.add_recurrent_connection(source, target)
        for target in self.region_order:
            self.mrnn.add_input_connection(spec.input_region_name, target)
        self.mrnn.finalize_connectivity()

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


__all__ = ["FixationMRNNModel", "FixationMRNNModelSpec", "build_model_spec"]
